# HOMIE integration decision

## Local `homie_g1_29dof`

The provided local deployment matches PATH's arm transport:

1. `deploy/robot_control/g1_wrapper.py` reads G1 state and publishes
   `rt/lowcmd`.
2. `deploy/arm_target_source.py` subscribes to `rt/arm_sdk` and reads the 14
   arm positions, gains, feed-forward torques, and enable marker.
3. `deploy/run_homie.py` combines those arm fields with the HOMIE policy's 15
   leg/waist targets. Its high-rate writer sends the merged 29-joint command to
   `rt/lowcmd`.

The current local arm source does not forward PATH's `dq` field into the merged
command: `G1RealWorldEnv.send_cmd()` sends zero target velocity. PATH's arm
positions, gains, and feed-forward torque are carried through; velocity
feed-forward is not.

Deploy this local repository on G1 with `arm_source: "arm_sdk"`. Run its HOMIE
controller separately from PATH-G1's `g1_bridge/run.sh`. Keep the HOMIE
controller as the only `rt/lowcmd` publisher.

## Upstream `HomieDeploy/unitree_sdk2`

The upstream [`g1_control.cpp`](https://github.com/InternRobotics/Homie/blob/main/HomieDeploy/unitree_sdk2/g1_control.cpp)
also writes combined 29-joint commands to `rt/lowcmd`, but it takes its inputs
from two different LCM channels: `pd_plustau_targets` for the 15 leg/waist
targets and `arm_action` for the 14 arm targets. It does not subscribe to
PATH's `rt/arm_sdk` output. The final Unitree topic is the same; the producer
and arm-input interface differ. The upstream writer therefore cannot directly
replace the provided local deployment while keeping PATH's current arm bridge
and protocol.

Do not start the upstream `g1_control` binary alongside local
`run_homie.py`; both publish `rt/lowcmd`. The upstream deployment's
[`LICENSE`](https://github.com/InternRobotics/Homie/blob/main/HomieDeploy/LICENSE)
also limits use to non-commercial purposes. PATH-G1 keeps HOMIE as a separate
deployment and does not copy its code into this repository.
