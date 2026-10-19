# Deployment layout

PATH-G1 runs the arm task and perception pipeline on Thor and its remote SAM3
worker. The G1 runs the SDK bridge in `g1_bridge/`, including Dex1 hand DDS
control. The local HOMIE/WBC is installed separately and is the sole
`rt/lowcmd` publisher.

## Repository packages

- `path_streaming/`: PICO inputs, PATH arm task references, and telemetry.
- `path_planning/`: nonlinear bimanual G1 arm IK.
- `perception/`: wrist D405 capture, SAM3 masks, surface geometry, belief
  assistance, and target publication.
- `assets/robot/`: pinned G1 URDF and all referenced meshes.
- `g1_bridge/`: LCM-to-Unitree arm/Dex1 bridge and G1 state publisher.
- `docs/`: control ownership, binary protocol, and deployment port registry.

The remote SAM3 inference worker is a separate host process. It contributes
camera masks and geometry over the documented LCM protocol; it never writes
robot actuator topics. See [`../perception/remote_offload/README.md`](../perception/remote_offload/README.md).

## Start order

1. On G1, start `g1_bridge/run.sh` in its default dry-run mode and confirm it
   receives `rt/lowstate` and publishes `G1_ROBOT_STATE`. Dry-run does not
   initialize or command the hands.
2. Start the Thor PATH state machine and the desired PICO input process.
3. Start the selected perception route. For wrist D405, start the Thor gateway
   and the remote SAM3 worker on the inference host. The Thor web server is an
   optional camera/mask view and does not publish targets unless explicitly
   enabled.
4. Start Unitree `dex1_1_service` on the computer connected to the hand serial
   cables, with a DDS interface that can communicate with G1, and leave it
   running. The live bridge waits up to five seconds for both
   `rt/dex1/*/state` streams. Install and launch commands are in
   [`g1_installation.md`](g1_installation.md).
5. Start the local `homie_g1_29dof` WBC on G1 with `arm_source: "arm_sdk"`.
   Confirm it is the only `rt/lowcmd` publisher.
6. Restart the G1 bridge with `G1_BRIDGE_LIVE=1` to forward PATH arm targets
   through `rt/arm_sdk` into the HOMIE command merger and route grip bits to
   the Dex1-1 service.

For the first camera/SAM3 integration run, keep the wrist gateway's end-effector
target publishing disabled. The checked-in wrist-to-camera transforms are
CAD-derived estimates, and the configured left/right camera serial assignment
is provisional. Calibrate both transforms and verify the physical camera sides
before enabling target-driven robot motion; see
[`../perception/remote_offload/README.md`](../perception/remote_offload/README.md).

Build details and bridge command line are in [`../g1_bridge/README.md`](../g1_bridge/README.md).
For the local-versus-upstream HOMIE comparison, see
[`homie_integration.md`](homie_integration.md).
For exact repository transfer, dependency installation, and run commands, see
[`g1_installation.md`](g1_installation.md).
