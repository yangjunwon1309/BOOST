# Deployment layout

PATH-G1 runs the arm task and perception pipeline on Thor. The G1 runs the
small SDK bridge in `g1_bridge/`. HOMIE/WBC and the Dex1 service remain
separate deployments.

## Repository packages

- `path_streaming/`: PICO inputs, PATH arm task references, and telemetry.
- `path_planning/`: nonlinear bimanual G1 arm IK.
- `perception/`: wrist D405 capture, SAM3 masks, surface geometry, belief
  assistance, and target publication.
- `assets/robot/`: pinned G1 URDF and all referenced meshes.
- `g1_bridge/`: LCM-to-Unitree `rt/arm_sdk` bridge and G1 state publisher.
- `docs/`: control ownership, binary protocol, and deployment port registry.

The remote SAM3 inference worker is a separate host process. It contributes
camera masks and geometry over the documented LCM protocol; it never writes
robot actuator topics. See [`../perception/remote_offload/README.md`](../perception/remote_offload/README.md).

## Start order

1. On G1, start `g1_bridge/run.sh` in its default dry-run mode and confirm it
   receives `rt/lowstate` and publishes `G1_ROBOT_STATE`.
2. Start the Thor PATH state machine and the desired PICO input process.
3. Start the selected perception route. For wrist D405, start the Thor gateway
   and the remote SAM3 worker on the inference host. The Thor web server is an
   optional camera/mask view and does not publish targets unless explicitly
   enabled.
4. Start the local `homie_g1_29dof` WBC on G1 with `arm_source: "arm_sdk"`.
   Confirm it is the only `rt/lowcmd` publisher.
5. Restart the G1 bridge with `G1_BRIDGE_LIVE=1` to forward PATH arm targets
   through `rt/arm_sdk` into the HOMIE command merger.

Build details and bridge command line are in [`../g1_bridge/README.md`](../g1_bridge/README.md).
For the local-versus-upstream HOMIE comparison, see
[`homie_integration.md`](homie_integration.md).
