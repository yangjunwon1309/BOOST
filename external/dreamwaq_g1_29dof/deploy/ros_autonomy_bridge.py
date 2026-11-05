#!/usr/bin/env python3
"""ROS1 Twist -> local DreamWaQ UDP; run with Noetic's system Python.

This module never imports Unitree DDS or constructs a motor publisher.
"""

import argparse
import json
import math
import select
import signal
import socket
import sys
import threading
import time
import uuid


class AutonomyBridgeState:
    def __init__(self, cmd_timeout=0.3, odom_timeout=1.0, clock=time.monotonic):
        for value in (cmd_timeout, odom_timeout):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("timeouts must be finite and positive")
        self.cmd_timeout = cmd_timeout
        self.odom_timeout = odom_timeout
        self.clock = clock
        self.lock = threading.Lock()
        self.hold = True
        self.damping = False
        self.command = (0.0, 0.0, 0.0)
        self.cmd_time = None
        self.raw_command = None
        self.raw_cmd_time = None
        self.odom_time = None
        self.session = uuid.uuid4().hex
        self.seq = 0

    def update_command(self, vx, vy, wz):
        values = tuple(float(v) for v in (vx, vy, wz))
        valid = all(math.isfinite(v) for v in values)
        with self.lock:
            now = self.clock()
            self.raw_command = values
            self.raw_cmd_time = now
            self.command = values if valid else (0.0, 0.0, 0.0)
            self.cmd_time = now if valid else None
        return valid

    def update_odometry(self):
        with self.lock:
            self.odom_time = self.clock()

    def set_hold(self, hold=None):
        with self.lock:
            self.hold = not self.hold if hold is None else bool(hold)
            # Releasing HOLD requires a new planner sample; don't replay a
            # command received while the operator was preparing the robot.
            self.command = (0.0, 0.0, 0.0)
            self.cmd_time = None
            return self.hold

    def request_damping(self):
        with self.lock:
            self.damping = True
            self.hold = True

    def packet(self, with_input=False):
        with self.lock:
            now = self.clock()
            command_fresh = (self.cmd_time is not None
                             and now - self.cmd_time <= self.cmd_timeout)
            odom_fresh = (self.odom_time is not None
                          and now - self.odom_time <= self.odom_timeout)
            state = ("damping" if self.damping else
                     "hold" if self.hold or not odom_fresh else "autonomy_active")
            command = self.command if state == "autonomy_active" and command_fresh else (0.0, 0.0, 0.0)
            reason = ("damping" if self.damping else "operator_hold" if self.hold else
                      "odometry_stale" if not odom_fresh else
                      "command_stale" if not command_fresh else "active")
            self.seq += 1
            packet = dict(vx=command[0], vy=command[1], wz=command[2],
                          seq=self.seq, session=self.session, state=state, reason=reason)
            if with_input:
                age_ms = None if self.raw_cmd_time is None else (now - self.raw_cmd_time) * 1000.0
                return packet, self.raw_command, age_ms
            return packet


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cmd-topic", default="/cmd/velocity")
    parser.add_argument("--odom-topic", default="/dreamstep_odom_g1")
    parser.add_argument("--align-state-topic", default="/align/state")
    parser.add_argument("--port", type=int, default=15002)
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--cmd-timeout", type=float, default=0.3)
    parser.add_argument("--odom-timeout", type=float, default=1.0)
    parser.add_argument("--no-keyboard", action="store_true")
    parser.add_argument("--print-commands", action="store_true",
                        help="print the latest ROS Twist and actual UDP command once per second, plus state changes")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in [1, 65535]")
    for name in ("rate", "cmd_timeout", "odom_timeout"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error("--%s must be finite and positive" % name.replace("_", "-"))
    return args


def main(argv=None):
    args = parse_args(argv)
    import termios
    import tty
    import rospy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from std_msgs.msg import Bool, Empty, String

    # ROS's synchronous signal handler can run while the main thread holds
    # a state/topic lock. Only mark a stop here; publish HOLD and clean up
    # after returning to the main loop.
    stopping = threading.Event()
    signal.signal(signal.SIGINT, lambda signum, frame: stopping.set())
    signal.signal(signal.SIGTERM, lambda signum, frame: stopping.set())
    rospy.init_node("dreamwaq_ros_autonomy_bridge", disable_signals=True)
    state = AutonomyBridgeState(args.cmd_timeout, args.odom_timeout)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    status_pub = rospy.Publisher("/dreamwaq/autonomy_status", String, queue_size=1, latch=True)

    def on_command(msg):
        if not state.update_command(msg.linear.x, msg.linear.y, msg.angular.z):
            rospy.logwarn_throttle(1.0, "non-finite planner command -> zero")

    def on_hold(msg):
        held = state.set_hold(msg.data)
        rospy.logwarn("DreamWaQ autonomy HOLD %s", "ON" if held else "OFF")

    def on_align(msg):
        if msg.data.strip() == "ALIGNED":
            state.set_hold(True)
            rospy.logwarn("alignment ALIGNED -> HOLD ON")

    subscribers = [
        rospy.Subscriber(args.cmd_topic, Twist, on_command, queue_size=1, tcp_nodelay=True),
        rospy.Subscriber(args.odom_topic, Odometry, lambda msg: state.update_odometry(), queue_size=1),
        rospy.Subscriber("/dreamwaq/hold", Bool, on_hold, queue_size=1),
        rospy.Subscriber("/dreamwaq/damping", Empty, lambda msg: state.request_damping(), queue_size=1),
        rospy.Subscriber(args.align_state_topic, String, on_align, queue_size=1),
    ]
    keyboard_fd = None
    old_terminal = None
    if not args.no_keyboard and sys.stdin.isatty():
        keyboard_fd = sys.stdin.fileno()
        old_terminal = termios.tcgetattr(keyboard_fd)
        tty.setcbreak(keyboard_fd)

    def send(packet):
        sock.sendto(json.dumps(packet, allow_nan=False, separators=(",", ":")).encode("utf-8"),
                    ("127.0.0.1", args.port))

    rospy.logwarn("HOLD ON: %s -> 127.0.0.1:%d; h=hold toggle, d=latched damping", args.cmd_topic, args.port)
    rospy.loginfo("fresh odometry required on %s; ALIGNED also enters HOLD", args.odom_topic)
    period = 1.0 / args.rate
    deadline = time.monotonic()
    last_reason = None
    last_print_reason = None
    next_print_time = 0.0
    try:
        while not stopping.is_set() and not rospy.is_shutdown():
            if keyboard_fd is not None and select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.read(1).lower()
                if key == "h":
                    state.set_hold()
                elif key == "d":
                    state.request_damping()
            if args.print_commands:
                packet, ros_input, input_age_ms = state.packet(with_input=True)
            else:
                packet = state.packet()
            send(packet)
            if args.print_commands and (time.monotonic() >= next_print_time
                                        or packet["reason"] != last_print_reason):
                if ros_input is None:
                    ros_text = "none"
                    age_text = "--"
                else:
                    ros_text = "vx={:+.3f} vy={:+.3f} wz={:+.3f}".format(*ros_input)
                    age_text = "{:.0f}ms".format(input_age_ms)
                print("[dwbridge] ROS {} in=({}) age={} | UDP 127.0.0.1:{} "
                      "out=(vx={:+.3f} vy={:+.3f} wz={:+.3f}) reason={} seq={}".format(
                          args.cmd_topic, ros_text, age_text, args.port,
                          packet["vx"], packet["vy"], packet["wz"],
                          packet["reason"], packet["seq"]), flush=True)
                last_print_reason = packet["reason"]
                next_print_time = time.monotonic() + 1.0
            if packet["reason"] != last_reason:
                last_reason = packet["reason"]
                rospy.logwarn("autonomy state: %s", last_reason)
            status_pub.publish(String(data=json.dumps(packet, allow_nan=False)))
            deadline += period
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            else:
                deadline = time.monotonic()
    finally:
        # Stopping this bridge requests zero/HOLD, not torque-off. Damping
        # requires d, the ROS damping topic, or Select at the real runtime.
        state.set_hold(True)
        for _ in range(3):
            try:
                send(state.packet())
            except OSError:
                break
        if old_terminal is not None:
            termios.tcsetattr(keyboard_fd, termios.TCSADRAIN, old_terminal)
        sock.close()
        for subscriber in subscribers:
            subscriber.unregister()
        rospy.signal_shutdown("autonomy bridge stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
