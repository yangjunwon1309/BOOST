"""Entry point for the policy process (see policy_ipc.py for why it's a process, not a
thread). Runs the HOMIE ONNX policy in a loop, talking to the I/O process purely through
PolicyIPC shared memory -- no DDS, no unitree_sdk2py, no env of any kind here.
"""
import time


def run(ipc, policy_config, stand_path, walk_path, onnx_threads, control_dt):
    from homie_policy import HomiePolicyV2

    policy = HomiePolicyV2(policy_config, stand_path, walk_path,
                            providers=["CPUExecutionProvider"], onnx_threads=onnx_threads)
    policy.reset()
    ipc.ready_event.set()
    print(f"[policy-proc] ready (onnx_threads={onnx_threads})")

    n = 0
    infer_ms_sum = infer_ms_max = 0.0
    last_print = time.time()
    try:
        while not ipc.stop_event.is_set():
            t0 = time.time()
            q29, dq29, quat, ang_vel, cmd = ipc.read_state()
            policy.set_command(cmd)

            t_i0 = time.time()
            q_target15, which = policy.update(q29, dq29, quat, ang_vel)
            infer_ms = (time.time() - t_i0) * 1000.0

            ipc.write_action(q_target15, which)

            n += 1
            infer_ms_sum += infer_ms
            infer_ms_max = max(infer_ms_max, infer_ms)
            if t0 - last_print > 1.0:
                # Periodic inference-performance logging disabled to keep the console quiet.
                # print(f"[policy-proc] [{which:>5}] infer_ms avg={infer_ms_sum / n:.2f} "
                #       f"max={infer_ms_max:.2f}  ({n} ticks/s)")
                n = 0
                infer_ms_sum = infer_ms_max = 0.0
                last_print = t0

            elapsed = time.time() - t0
            if elapsed < control_dt:
                time.sleep(control_dt - elapsed)
    except KeyboardInterrupt:
        pass
    print("[policy-proc] stopped")
