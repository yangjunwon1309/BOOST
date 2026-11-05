"""Check the native API needed to return from user control without restarting.

Run while already in native walking, with DreamWaQ stopped. No LowCmd publisher
is created. A successful result only checks the return API; it does not establish
that user control or a continuous-support handoff works.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

try:
    from .controller_handoff import MotionServiceError, UnitreeMotionService, _check_code
except ImportError:
    from controller_handoff import MotionServiceError, UnitreeMotionService, _check_code


RPC_SERVER_API_NOT_IMPLEMENTED = 3203
WALKRUN = 2  # SDK InternalFsmMode.WALKRUN (API 7111 parameter data=2)


def require_dreamwaq_stopped(proc_root=Path("/proc"), own_pid=None):
    """Do not change native ownership while a DreamWaQ motor writer is active."""
    own_pid = os.getpid() if own_pid is None else own_pid
    entrypoints = {"dreamwaq-g1-29dof", "run_dreamwaq.py", "homie_base_dreamwaq.py",
                   "deploy.run_dreamwaq", "dreamwaq_interface.scripts.homie_base_dreamwaq"}
    for entry in proc_root.iterdir():
        if not entry.name.isdigit() or int(entry.name) == own_pid:
            continue
        try:
            args = entry.joinpath("cmdline").read_bytes().decode().split("\0")
        except (OSError, UnicodeError):
            continue
        if any(Path(arg).name in entrypoints for arg in args if arg):
            raise MotionServiceError("DreamWaQ is running (PID %s); stop it before probing"
                                     % entry.name)


def check_live_return_api(service, guard=require_dreamwaq_stopped):
    """Request WALKRUN only when native walking is already active.

    Do not start/restart native AI, enter user control, or send SetFsmId. In
    particular, reject a response timeout instead of using an unchanged walking
    FSM as evidence that an unacknowledged API is supported.
    """
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "api": "SwitchToInternalCtrl(WALKRUN)",
        "api_id": 7111,
        "status": "not_checked",
        "continuous_support_validated": False,
    }
    try:
        guard()
        report["controller"] = service.check_mode()
        if not report["controller"]:
            raise MotionServiceError("native controller is stopped; prepare native walking first")
        report["fsm_before"] = service._fsm()
        if report["fsm_before"] not in (500, 501):
            raise MotionServiceError("native FSM %s is not walking; no transition sent"
                                     % report["fsm_before"])
        guard()
        method = getattr(service.loco, "SwitchToInternalCtrl", None)
        if not callable(method):
            report["status"] = "client_api_missing"
            raise MotionServiceError("installed SDK has no SwitchToInternalCtrl")
        result = method(WALKRUN)
        report["rpc_code"] = result[0] if isinstance(result, tuple) else result
        _check_code(result, report["api"])
        report["fsm_after"] = service._fsm()
        if report["fsm_after"] not in (500, 501):
            raise MotionServiceError("return API acknowledged, but native walking was not confirmed")
        report["status"] = "return_api_acknowledged"
    except MotionServiceError as exc:
        report["error"] = str(exc)
        if exc.code == RPC_SERVER_API_NOT_IMPLEMENTED:
            report["status"] = "server_api_not_implemented"
            try:
                report["fsm_after"] = service._fsm()
            except MotionServiceError as read_error:
                report["fsm_after_error"] = str(read_error)
        elif report["status"] == "not_checked":
            report["status"] = "check_failed"
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--net", required=True)
    parser.add_argument("--report", type=Path,
                        help="write the observed result as JSON")
    args = parser.parse_args(argv)
    require_dreamwaq_stopped()
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize

    ChannelFactoryInitialize(0, args.net)
    service = UnitreeMotionService.connect(cancel=require_dreamwaq_stopped)
    report = check_live_return_api(service)
    encoded = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded)
    print(encoded, end="")
    return 0 if report["status"] == "return_api_acknowledged" else 1


if __name__ == "__main__":
    raise SystemExit(main())
