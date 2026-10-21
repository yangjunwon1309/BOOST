# Deployment layout

BOOST keeps controller, robot I/O, and perception services in separate
packages. The current runtime pieces were carried over from PATH-G1; this
deployment map records where they live after the repository reorganization.

- `boost/teleop/`: XR managers, PATH state machine, teleop messages, and
  isolated legacy intention modules.
- `boost/io/path_transport/`: PATH LCM packets and ZMQ transport helpers.
- `boost/control/arm/path_planning/`: nonlinear bimanual G1 arm IK.
- `boost/io/g1_bridge/`: Unitree arm SDK and legacy Dex1 hand bridge.
- `services/pose_service/`: wrist D405 capture, remote SAM3, surface geometry,
  and the existing target publisher.
- `external/homie_g1_29dof/`: local HOMIE/WBC deployment snapshot. It remains
  the sole `rt/lowcmd` writer; BOOST's G1 bridge ends at `rt/arm_sdk`.
- `assets/robot/`: G1 model and BrainCo Revo2 model assets.

## Current runtime boundary

The G1 bridge and its protocol still use PATH's 14-joint arm packet and Dex1
open/close flags. They are not a Revo2 driver. The Revo2 service uses separate
left/right six-joint command and state topics; BOOST's adapter is future work.
Likewise, the current wrist service is SAM3/surface based. FoundationPose,
BOOST's pose schema, recording pipeline, and object-aware MPPI remain planned
in [`../BOOST_dev_plan.md`](../BOOST_dev_plan.md).

## Existing PATH launch sequence

The current PATH-derived XR and state-machine processes can be launched from
the BOOST repository root:

1. Start the G1 bridge in dry-run mode on G1 with
   `boost/io/g1_bridge/run.sh`, then confirm the `G1_ROBOT_STATE` stream.
2. Start `scripts/run_pico_manager.sh` or
   `scripts/run_galaxy_xr_manager.sh` for the selected XR input.
3. Start `scripts/run_state_machine_bridge.sh` for the PATH teleop bridge.
4. Start the wrist capture and remote SAM3 processes from
   `services/pose_service/` when checking the existing perception pipeline.
5. Run the local HOMIE/WBC separately as the only `rt/lowcmd` publisher.

The existing Dex1 live route requires Unitree's `dex1_1_service` and must not
be treated as the Revo2 route. The old PATH/Dex1 installation details remain in
[`g1_installation.md`](g1_installation.md); the interface contracts are in
[`protocols.md`](protocols.md) and [`control_ownership.md`](control_ownership.md).
