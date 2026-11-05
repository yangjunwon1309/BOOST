"""Sequence-coupled isolated DreamWaQ policy process."""

import os
import time

import numpy as np

from dreamwaq_policy_ipc import IPCLockTimeout, PolicyActionResponse


def run(
    ipc,
    export_path,
    device,
    control_dt,
    policy_cpu=None,
    policy_prewake_us=8000.0,
    clip_lower_joint_targets=True,
):
    """Infer exactly once for each newly published parent state.

    ``control_dt`` bounds the wakeup poll used during shutdown. The worker has
    no independent 50-Hz sleep: the parent's state request is the policy clock.
    """

    from dreamwaq_policy import DreamWaQPolicy

    if policy_cpu is not None:
        os.sched_setaffinity(0, {int(policy_cpu)})
    policy = DreamWaQPolicy(
        export_path,
        device=device,
        clip_lower_joint_targets=clip_lower_joint_targets,
    )
    if ipc.action_dim != policy.contract.action_dim:
        raise RuntimeError(
            "IPC action dimension %d differs from export dimension %d"
            % (ipc.action_dim, policy.contract.action_dim)
        )
    ipc.ready_event.set()
    print("[policy-proc] DreamWaQ sequence-coupled worker ready", flush=True)

    last_sequence = -1
    last_policy_epoch = None
    skipped_total = 0
    previous_request_published_ns = None
    control_period_ns = int(float(control_dt) * 1.0e9)
    prewake_ns = int(max(0.0, float(policy_prewake_us)) * 1.0e3)
    try:
        while not ipc.stop_event.is_set():
            spin_after_ns = None
            if previous_request_published_ns is not None and prewake_ns > 0:
                spin_after_ns = (
                    previous_request_published_ns + control_period_ns - prewake_ns
                )
            request = ipc.wait_for_state(
                last_sequence,
                timeout_s=max(0.02, min(0.1, float(control_dt))),
                spin_after_ns=spin_after_ns,
            )
            if request is None:
                # Native control can own the robot for minutes. Do not busy
                # spin against the final DreamWaQ tick throughout that pause.
                previous_request_published_ns = None
                continue
            # Across a new epoch the gap is the other export's worker's states
            # (policy_switch.py), not ones this worker missed.
            if (last_sequence >= 0 and request.policy_epoch == last_policy_epoch
                    and request.sequence > last_sequence + 1):
                skipped_total += request.sequence - last_sequence - 1
            last_sequence = request.sequence
            previous_request_published_ns = request.request_published_ns

            started_ns = time.monotonic_ns()
            # The first update already seeds the policy from measured state.
            # Only a later controller session needs an explicit reset.
            if last_policy_epoch is not None and request.policy_epoch != last_policy_epoch:
                policy.reset(request.q29)
            last_policy_epoch = request.policy_epoch
            policy.set_command(request.command)
            policy.set_locomotion_mode(request.locomotion_mode)
            q_target15, which, inference_ns = policy.update(
                request.q29,
                request.dq29,
                request.quaternion_wxyz,
                request.angular_velocity_body,
                previous_lower_action=request.previous_lower_action,
            )
            finished_ns = time.monotonic_ns()
            try:
                ipc.publish_action(
                    PolicyActionResponse(
                        source_state_sequence=request.sequence,
                        source_state_monotonic_ns=request.monotonic_ns,
                        source_state_published_ns=request.request_published_ns,
                        inference_started_ns=started_ns,
                        inference_finished_ns=finished_ns,
                        skipped_state_count=skipped_total,
                        q_target15=q_target15,
                        normalized_lower_action=policy.last_lower_action.copy(),
                        mode=which,
                        inference_ms=float(inference_ns) / 1.0e6,
                        max_action=float(np.max(np.abs(policy.last_lower_action))),
                    )
                )
            except IPCLockTimeout:
                # The parent held the action lock past the deadline (scheduling
                # delay). Drop this action; the parent counts it as a miss.
                continue
    except KeyboardInterrupt:
        pass
    finally:
        policy.close()
        print("[policy-proc] stopped", flush=True)
