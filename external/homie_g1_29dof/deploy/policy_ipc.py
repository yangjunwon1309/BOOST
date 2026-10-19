"""Shared-memory IPC between run_homie.py (I/O process: DDS state/lowcmd/arm_sdk) and
homie_policy_process.py (policy process: onnxruntime inference only).

Why a second process instead of a second thread: CPython's GIL serializes ALL Python
bytecode execution within one process, across every thread. Measured on the real G1
Jetson Orin NX (even with CPU clocks pinned via `jetson_clocks`, ruling out DVFS ramp
latency), onnxruntime inference inside the same process as the rt/lowcmd writer thread
was still observed to stall that writer for 10-50ms bursts, throttling teleop arm
tracking well below xr_teleoperate's own ~250 Hz rt/arm_sdk rate. A separate OS process
has its own GIL, so inference time -- however long -- can never block the writer/reader
threads in the I/O process.

'Latest wins' semantics, same philosophy as arm_target_source.py: each side always reads
whatever the other last wrote, tolerating a few ms of staleness rather than blocking.
Two independent locks (state vs action) so a slow policy tick can never make the I/O
process wait on a lock either.
"""
import multiprocessing as mp

import numpy as np

STATE_DIM = 29 + 29 + 4 + 3 + 3  # q29, dq29, quat(wxyz), ang_vel, cmd[vx,vy,wz]
ACTION_DIM = 15 + 1              # q_target15, which_code (0.0=stand, 1.0=walk)


class PolicyIPC:
    def __init__(self, ctx=None):
        """`ctx` must be the SAME multiprocessing context used to spawn the policy
        process (e.g. `multiprocessing.get_context("spawn")`). Creating these
        synchronization primitives from the default context while launching the child
        under a different one is a known source of the child hanging silently during
        its own startup -- the Array/Event's semaphores get registered against the
        wrong context."""
        ctx = ctx or mp
        self.state = ctx.Array('f', STATE_DIM, lock=True)
        self.action = ctx.Array('f', ACTION_DIM, lock=True)
        self.stop_event = ctx.Event()
        self.ready_event = ctx.Event()  # set once the policy process has loaded its ONNX models

    # --- I/O process side ---------------------------------------------------
    def write_state(self, q29, dq29, quat, ang_vel, cmd):
        with self.state.get_lock():
            buf = np.frombuffer(self.state.get_obj(), dtype=np.float32)
            buf[0:29] = q29
            buf[29:58] = dq29
            buf[58:62] = quat
            buf[62:65] = ang_vel
            buf[65:68] = cmd

    def read_action(self):
        with self.action.get_lock():
            buf = np.frombuffer(self.action.get_obj(), dtype=np.float32).copy()
        return buf[0:15].copy(), ("walk" if buf[15] >= 0.5 else "stand")

    # --- policy process side -------------------------------------------------
    def read_state(self):
        with self.state.get_lock():
            buf = np.frombuffer(self.state.get_obj(), dtype=np.float32).copy()
        return buf[0:29], buf[29:58], buf[58:62], buf[62:65], buf[65:68]

    def write_action(self, q_target15, which):
        with self.action.get_lock():
            buf = np.frombuffer(self.action.get_obj(), dtype=np.float32)
            buf[0:15] = q_target15
            buf[15] = 1.0 if which == "walk" else 0.0
