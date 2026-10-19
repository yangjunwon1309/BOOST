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

The deployment source to install is the local
`C:\studioprojects\homie_g1_29dof\homie_g1_29dof` folder. It has no Git remote;
the Windows-to-G1 transfer and dependency commands are in
[`g1_installation.md`](g1_installation.md).

## How to track the local snapshot

A branch is a normal way to stage and review a change, but it is not a separate
repository or a durable link to Homie's upstream history. Because this local
folder has no `.git` metadata, copying it to a branch in PATH-G1 would create a
PATH-G1 commit containing a snapshot. If provenance and redistribution terms
are established, a fixed self-contained deployment snapshot can live under a
clear path such as `external/homie_g1_29dof/`; a `homie-g1-deploy` branch can
stage and review that addition before merging it into `deploy`.

The local folder currently has no `LICENSE` file, and its missing Git history
means its exact upstream base cannot be verified from the checkout. The
upstream [`HomieDeploy/LICENSE`](https://github.com/InternRobotics/Homie/blob/main/HomieDeploy/LICENSE)
states CC BY-NC 4.0. If this local code derives from that deployment, keep its
required attribution/license notices and account for the non-commercial limit
before publishing or redistributing the snapshot. Do not infer a license from
the PATH-G1 repository's license.

For ongoing upstream updates, the usual maintainable arrangement is to retain
Homie as its own Git repository (typically a fork that preserves its upstream
remote/history) and pin the required commit from PATH-G1 with a submodule or
subtree. A submodule keeps the repositories independent and pins a commit, but
requires recursive initialization when cloning. A subtree includes the source
in an ordinary clone and can still pull upstream history. First recover or
establish the local Homie's true Git ancestry before choosing either; simply
running `git init` on this copy would not reconnect it to `InternRobotics/Homie`.

Once the exact base and license are confirmed, either a reviewed vendored
snapshot is suitable for a fixed deploy bundle, or a fork/submodule is better
for ongoing upstream updates. Until then, a PATH-G1 review branch can preserve
the proposed layout but does not resolve source history or redistribution
terms. Do not use a PATH-G1 branch as a permanent substitute for an independent
Homie repository.

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
