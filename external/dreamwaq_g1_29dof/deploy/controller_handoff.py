"""Exclusive ownership handoff between rt/lowcmd and Unitree's AI controller.

This uses the existing Develop-mode transport, not rt/user_lowcmd. SDK imports
are lazy so transition ordering can be exercised without DDS or a robot.
"""

from __future__ import annotations

import time


CONTROLLER_TOGGLE_MASK = 0x0002 | 0x0800  # L1 + Y (no built-in motion shortcut)
RPC_SEND_ERROR = 3102
RPC_TIMEOUT = 3104
TRANSIENT_READ_ERRORS = (RPC_SEND_ERROR, RPC_TIMEOUT)
LOCO_STATE_NOT_AVAILABLE = 7301
LOCO_READ_RETRY_ERRORS = (*TRANSIENT_READ_ERRORS, LOCO_STATE_NOT_AVAILABLE)
SWITCHER_BUSY = 7002


class MotionServiceError(RuntimeError):
    def __init__(self, message, *, code=None):
        super().__init__(message)
        self.code = code


def _check_code(result, operation):
    code = result[0] if isinstance(result, tuple) else result
    if code != 0:
        raise MotionServiceError("%s failed: SDK code=%s" % (operation, code), code=code)
    return result


class UnitreeMotionService:
    def __init__(self, switcher, loco, *, timeout_s=8.0, cancel=lambda: None,
                 notify=print):
        self.switcher = switcher
        self.loco = loco
        self.timeout_s = float(timeout_s)
        self.cancel = cancel
        self.notify = notify

    @classmethod
    def connect(cls, **kwargs):
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
        from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

        switcher, loco = MotionSwitcherClient(), LocoClient()
        for client in (switcher, loco):
            client.SetTimeout(1.0)
            client.Init()
        return cls(switcher, loco, **kwargs)

    def check_mode(self):
        self.cancel()
        result = _check_code(self.switcher.CheckMode(), "CheckMode")
        data = result[1]
        if not isinstance(data, dict) or not isinstance(data.get("name"), str):
            raise MotionServiceError("CheckMode returned an invalid controller name")
        return data["name"]

    def _poll(self, read, accept, label, *, retry_codes=TRANSIENT_READ_ERRORS):
        deadline = time.monotonic() + self.timeout_s
        last_error = None
        initializing_reported = False
        while True:
            self.cancel()
            try:
                value = read()
                if accept(value):
                    return value
                last_error = None
            except MotionServiceError as exc:
                # A native-controller restart may temporarily interrupt RPCs.
                # Loco reads also allow 7301 while native LocoState initializes.
                # Other server errors and invalid response data remain fatal.
                if exc.code not in retry_codes:
                    raise
                last_error = exc
                if exc.code == LOCO_STATE_NOT_AVAILABLE and not initializing_reported:
                    self.notify("[controller] Unitree locomotion state is initializing "
                                "(7301); waiting for state readiness")
                    initializing_reported = True
            if time.monotonic() >= deadline:
                detail = "; last RPC error: %s" % last_error if last_error else ""
                raise MotionServiceError("timed out waiting for %s%s" % (label, detail))
            time.sleep(0.2)

    def _wait(self, predicate, label, *, retry_codes=TRANSIENT_READ_ERRORS):
        return self._poll(predicate, bool, label, retry_codes=retry_codes)

    def _request(self, call, operation, *, retry_busy=False):
        deadline = time.monotonic() + self.timeout_s if retry_busy else None
        busy_reported = False
        while True:
            self.cancel()
            try:
                _check_code(call(), operation)
                return 0
            except MotionServiceError as exc:
                if retry_busy and exc.code == SWITCHER_BUSY:
                    # Busy explicitly rejects the request. Retry within the
                    # deadline, including an exit requested during AI startup.
                    if time.monotonic() >= deadline:
                        raise MotionServiceError("timed out waiting for %s; %s"
                                                 % (operation, exc), code=exc.code) from exc
                    if not busy_reported:
                        self.notify("[controller] %s: switcher busy (7002); waiting"
                                    % operation)
                        busy_reported = True
                    time.sleep(0.2)
                    continue
                if exc.code != RPC_TIMEOUT:
                    raise
                # 3104 means the response timed out, not that the command failed.
                # Send once, then require the resulting state before changing owner.
                self.notify("[controller] %s response timed out (3104); "
                            "waiting for state confirmation" % operation)
                return RPC_TIMEOUT

    def release(self):
        # CheckMode can still say 'ai' while its remote has selected Develop.
        # Explicitly stop that service before either acquiring low-level control
        # or restarting native control; name alone is not a Develop-mode flag.
        mode = self._poll(self.check_mode, lambda _: True, "Unitree controller status")
        if mode:
            self._request(self.switcher.ReleaseMode, "ReleaseMode", retry_busy=True)
        self._wait(lambda: not self.check_mode(), "Unitree controller release")

    def select(self, name):
        self._request(lambda: self.switcher.SelectMode(name), "SelectMode(%s)" % name,
                      retry_busy=True)
        self._wait(self.check_mode, "Unitree controller startup")

    def _fsm(self):
        self.cancel()
        result = _check_code(self.loco.GetFsmId(), "GetFsmId")
        if not isinstance(result[1], int):
            raise MotionServiceError("GetFsmId returned an invalid FSM ID")
        return result[1]

    def start_walking(self):
        # Use SetFsmId directly: several Python SDK convenience methods discard
        # the RPC return code. Start() requests FSM 500. On this G1, a freshly
        # started AI controller in FSM 0 ignores both 4 and 500 until FSM 1.
        fsm = self._poll(self._fsm, lambda _: True, "locomotion state readiness",
                         retry_codes=LOCO_READ_RETRY_ERRORS)
        if fsm in (500, 501):
            return
        if fsm == 0:
            self.notify("[controller] preparing native damping (FSM 0 -> 1)")
            self._request(lambda: self.loco.SetFsmId(1), "native damping (FSM 1)")
            self._wait(lambda: self._fsm() == 1, "native damping",
                       retry_codes=LOCO_READ_RETRY_ERRORS)
        if fsm in (0, 1):
            self.notify("[controller] preparing native locked standing (FSM 1 -> 4)")
            self._request(lambda: self.loco.SetFsmId(4), "locked standing (FSM 4)")
            self._wait(lambda: self._fsm() == 4, "locked standing",
                       retry_codes=LOCO_READ_RETRY_ERRORS)
            fsm = 4
        self.notify("[controller] requesting Unitree regular walking directly "
                    "(FSM %s -> 500)" % fsm)
        request_deadline = time.monotonic() + self.timeout_s
        outcome = self._request(lambda: self.loco.SetFsmId(500), "regular walking (FSM 500)")

        def walking_confirmed():
            nonlocal outcome
            current = self._fsm()
            if current in (500, 501):
                return True
            # FSM 4 is reported before its physical preparation finishes. This
            # firmware can acknowledge 500 while remaining in 4; renew that
            # same target only after a successful response and an observed 4.
            # An RPC timeout leaves an unknown outcome and is never resent.
            if current == 4 and outcome == 0 and time.monotonic() < request_deadline:
                outcome = self._request(lambda: self.loco.SetFsmId(500),
                                        "regular walking (FSM 500)")
            return False

        self._wait(walking_confirmed, "regular walking",
                   retry_codes=LOCO_READ_RETRY_ERRORS)


class ControllerHandoff:
    def __init__(self, env, service, *, native_mode="ai", notify=print,
                 initial_keys=0):
        self.env = env
        self.service = service
        self.native_mode = native_mode
        self.notify = notify
        self.controller = "unclaimed"
        self.walking_confirmed = False
        self._pressed = (int(initial_keys) & CONTROLLER_TOGGLE_MASK) == CONTROLLER_TOGGLE_MASK

    def toggle_requested(self, keys):
        keys = int(keys)
        pressed = (keys & CONTROLLER_TOGGLE_MASK) == CONTROLLER_TOGGLE_MASK
        requested = keys == CONTROLLER_TOGGLE_MASK and pressed and not self._pressed
        self._pressed = pressed
        return requested

    def enter_dreamwaq(self):
        self.controller = "switching"
        self.notify("[controller] releasing Unitree controller for DreamWaQ")
        self.service.release()
        self.env.require_fresh_state()
        self.env.set_lowcmd_enabled(True)
        self.controller = "dreamwaq"
        self.walking_confirmed = False

    def enter_unitree(self):
        self.controller = "switching"
        self.notify("[controller] stopping DreamWaQ LowCmd writer")
        self.env.suspend_lowcmd()
        # Restart even if CheckMode already reports 'ai': it may be in Develop.
        self.service.release()
        self.service.select(self.native_mode)
        self.controller = "unitree"
        self.walking_confirmed = False
        self.notify("[controller] Unitree owns control; DreamWaQ LowCmd is disabled")
        try:
            self.service.start_walking()
        except MotionServiceError as exc:
            self.notify(
                "[controller] automatic walking unavailable: %s; use the remote "
                "L2+UP (locked standing), then R1+X (regular walking). "
                "L1+Y returns to DreamWaQ; Select exits to damping." % exc
            )
        else:
            self.walking_confirmed = True
            self.notify("[controller] Unitree regular walking confirmed")

    def prepare_shutdown(self):
        # Never send even the final damping LowCmd over the native controller.
        # Disable cancellation here: Select is exactly why cleanup was requested.
        cancel = self.service.cancel
        self.service.cancel = lambda: None
        try:
            if self.controller != "dreamwaq":
                self.env.suspend_lowcmd()
                self.service.release()
                self.env.require_fresh_state()
                self.env.set_lowcmd_enabled(True)
                self.controller = "dreamwaq"
        finally:
            self.service.cancel = cancel
