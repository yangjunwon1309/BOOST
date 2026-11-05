"""LEGACY bridge: subscribe the network `rt/arm_sdk` DDS topic (published by
xr_teleoperate on a separate board, e.g. a Thor jetson) and republish the 14 arm targets
to Redis.

`run_homie.py` no longer needs this: `robot_control/g1_wrapper.py` uses `unitree_sdk2py`
exclusively (not `unitree_interface`), so `arm_source: "arm_sdk"` in
`robot_control/configs/g1_homie.yaml` subscribes `rt/arm_sdk` directly, in the same
process, with no DDS-binding conflict. Prefer that (the default) unless you have another
reason to decouple the arm feed over Redis into a separate process.

Usage:
    python arm_sdk_to_redis.py --net eth0
"""
import argparse
import json
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as hg_LowCmd

ARM_MOTOR_IDS = list(range(15, 29))  # G1_29_JointArmIndex, hardware order
ARM_SDK_TOPIC = "rt/arm_sdk"
ARM_SDK_WEIGHT_ID = 29  # kNotUsedJoint0 carries the enable weight
ARM_SDK_WEIGHT_ON = 0.5
REDIS_KEY = "action_arm_unitree_g1_homie"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True, help="network interface, e.g. eth0")
    ap.add_argument("--redis-host", default="localhost")
    ap.add_argument("--redis-port", type=int, default=6379)
    ap.add_argument("--redis-key", default=REDIS_KEY)
    ap.add_argument("--rate", type=float, default=50.0, help="publish rate (Hz)")
    args = ap.parse_args()

    import redis

    ChannelFactoryInitialize(0, args.net)
    sub = ChannelSubscriber(ARM_SDK_TOPIC, hg_LowCmd)
    sub.Init()
    print(f"[bridge] subscribed '{ARM_SDK_TOPIC}' on {args.net}, "
          f"publishing to redis://{args.redis_host}:{args.redis_port} key='{args.redis_key}'")

    client = redis.Redis(host=args.redis_host, port=args.redis_port, db=0)
    period = 1.0 / args.rate
    last_weight = None

    while True:
        msg = sub.Read()
        if msg is not None:
            weight = msg.motor_cmd[ARM_SDK_WEIGHT_ID].q
            if weight >= ARM_SDK_WEIGHT_ON:
                arm = [float(msg.motor_cmd[i].q) for i in ARM_MOTOR_IDS]
                client.set(args.redis_key, json.dumps({"arm": arm, "t": time.time()}))
            if weight != last_weight:
                print(f"[bridge] teleop weight={weight:.2f} "
                      f"({'active' if weight >= ARM_SDK_WEIGHT_ON else 'idle'})")
                last_weight = weight
        time.sleep(period)


if __name__ == "__main__":
    main()
