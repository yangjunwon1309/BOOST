# HOMIE integration decision

## Local `homie_g1_29dof`

The included snapshot at [`external/homie_g1_29dof/`](../external/homie_g1_29dof/)
matches PATH's arm transport:

1. `deploy/robot_control/g1_wrapper.py` reads G1 state and publishes
   `rt/lowcmd`.
2. `deploy/arm_target_source.py` subscribes to `rt/arm_sdk` and reads the 14
   arm positions, gains, feed-forward torques, and enable marker.
3. `deploy/run_homie.py` combines those arm fields with the HOMIE policy's 15
   leg/waist targets. Its high-rate writer sends the merged 29-joint command to
   `rt/lowcmd`.

The arm source does not forward PATH's `dq` field into the merged command:
`G1RealWorldEnv.send_cmd()` sends zero target velocity. PATH's arm positions,
gains, and feed-forward torque are carried through; velocity feed-forward is
not.

Deploy this snapshot on G1 with `arm_source: "arm_sdk"`. Run its HOMIE
controller separately from PATH-G1's `g1_bridge/run.sh`. Keep the HOMIE
controller as the only `rt/lowcmd` publisher. G1 transfer and dependency
commands are in [`g1_installation.md`](g1_installation.md).

## Snapshot and license

The customized local deployment is packaged as a source snapshot in
`external/homie_g1_29dof/`. It has no Git metadata, so its exact upstream
commit and complete change history are unknown. The source attribution,
comparison link, known local changes, and provenance limitation are recorded
in [`SOURCE.md`](../external/homie_g1_29dof/SOURCE.md).

The HOMIE root README and
[`HomieDeploy/README.md`](https://github.com/InternRobotics/Homie/blob/main/HomieDeploy/README.md)
declare CC BY-NC-SA 4.0. The upstream
[`HomieDeploy/LICENSE`](https://github.com/InternRobotics/Homie/blob/main/HomieDeploy/LICENSE)
currently states CC BY-NC 4.0, so the upstream notices conflict. This snapshot
follows the CC BY-NC-SA 4.0 README declaration; its scoped notice is in
[`external/homie_g1_29dof/LICENSE`](../external/homie_g1_29dof/LICENSE).
That license applies to the HOMIE-derived snapshot only. It does not apply to
the rest of PATH-G1 or separately identified third-party components. Preserve
attribution, identify modifications, and follow the non-commercial and
ShareAlike terms when redistributing this snapshot. The PATH-G1 root license
does not change the snapshot's terms.

For ongoing upstream updates, retain Homie as its own Git repository and pin a
required commit from PATH-G1 with a submodule or subtree. Initializing Git in
the supplied copy would not reconnect it to `InternRobotics/Homie`.

## Upstream `HomieDeploy/unitree_sdk2`

The upstream
[`g1_control.cpp`](https://github.com/InternRobotics/Homie/blob/main/HomieDeploy/unitree_sdk2/g1_control.cpp)
also writes combined 29-joint commands to `rt/lowcmd`, but it takes its inputs
from two different LCM channels: `pd_plustau_targets` for the 15 leg/waist
targets and `arm_action` for the 14 arm targets. It does not subscribe to
PATH's `rt/arm_sdk` output. The final Unitree topic is the same; the producer
and arm-input interface differ. The upstream writer therefore cannot directly
replace the customized local deployment while keeping PATH's current arm
bridge and protocol.

Do not start the upstream `g1_control` binary alongside local
`run_homie.py`; both publish `rt/lowcmd`. The upstream
`HomieDeploy/README.md` declares CC BY-NC-SA 4.0, while its
[`LICENSE`](https://github.com/InternRobotics/Homie/blob/main/HomieDeploy/LICENSE)
says CC BY-NC 4.0. PATH-G1 includes its customized deployment snapshot under
`external/homie_g1_29dof/` with the scoped license notice described above.
