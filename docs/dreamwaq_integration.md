# DreamWaQ integration

## Runtime boundary

DreamWaQ is the sole hardware `rt/lowcmd` writer. It can receive arm targets
from either of two mutually exclusive sources:

1. The existing PATH flow sends 14 arm targets as PDU1 over LCM; the G1 arm
   adapter republishes them on `rt/arm_sdk`.
2. The direct, arm-only Galaxy XR flow in
   [`g1_xr_arm_teleop.md`](g1_xr_arm_teleop.md) runs Televuer and G1_29 IK on
   Thor, publishing `rt/arm_sdk` directly.

Both use the same `arm_source: arm_sdk` DreamWaQ reader and G1 29-joint
contract. Do not run the PATH adapter and direct XR publisher together: both
publish `rt/arm_sdk`. Revo2 MPPI is a separate optional hand route in the PATH
flow; the direct arm-only runner does not start it or publish finger commands.

The selected deployment source is `url_g1_deploy` commit
`3d37aa43f88964fc4255857c840dd33dd08e6eda`. The default model/export pair is
`new2_3_backpack_lower8cm_299999`: its manifest records a 12-action lower-body
policy, 99-dimensional observation, and no explicit standing/walking mode
input. Its backpack training domain is 2–5 kg. The MuJoCo model adds the
nominal 3.5 kg box at the manifest center `[-0.12464276, 0, 0.1]` m with
`[0.11, 0.18, 0.34]` m dimensions. The runtime export and payload receipt are
in `external/dreamwaq_g1_29dof/checkpoints/new2_3_backpack_lower8cm_299999/exported/`;
the earlier `new3_4_backpack_440000` export remains available as an alternate.
The manifest and policy files are hash-recorded in `external/dreamwaq_g1_29dof/SOURCE.md`.
The original training checkpoint is not needed to run inference.

The deployment config and action compiler retain the target joint order,
waist behavior, gains, torque limits, command timing, deadline handling, and
arm-target smoothing. BOOST does not layer additional reference filtering or a
backpack compensation torque onto that runtime. The selected export manifest
and target deployment config govern those values.

## G1 hardware launch: PATH adapter flow

The DreamWaQ and Teleimager Python 3.10 environments are installed separately
outside the repository, under `$HOME/.local/share/boost/external/venvs/` by
default. Override `BOOST_EXTERNAL_ROOT` or the individual `BOOST_*_ENV_ROOT`
variables in `configs/teleop/local.env` to choose another install location.

```bash
bash scripts/setup_dreamwaq_runtime.sh
bash scripts/setup_teleimager_runtime.sh
```

Build the arm/state adapter against Unitree SDK2 C++ as described in
[`../boost/io/g1_bridge/README.md`](../boost/io/g1_bridge/README.md). From the
G1 checkout, start the one camera server and the body controller separately:

```bash
bash scripts/run_g1_teleimager.sh

BOOST_ALLOW_HARDWARE_CONTROL=1 \
G1_NETWORK_INTERFACE=eth0 \
bash scripts/run_g1_dreamwaq_controller.sh
```

The hardware launcher starts the PATH arm/state adapter, then invokes the
target `deploy/run_dreamwaq.py` with the selected export, 15 ms action deadline,
and three-timeout diagnostic window. It intentionally requires the explicit
`BOOST_ALLOW_HARDWARE_CONTROL=1` environment setting. The target runtime still
requires its START then A enable procedure. The default NEW2-3 export has no
explicit standing/walking mode input, so the X mode toggle is inactive. Do not
start a second body controller or `g1_ws/g1_lowlevel_bridge` alongside it.
The `g1_ws` bridge publishes `rt/lowcmd` even in its null-action smoke mode.

The target runner's default CPU partition reserves camera cores 0–2, DDS I/O
cores 3–5, and selects policy/control cores from the remaining CPUs. The
Teleimager launcher uses camera cores 0–2. Check these masks against the G1's
available cores before deployment; change the target runner's CPU options and
`G1_TELEIMAGER_CAMERA_CPUSET` together if the layout differs.

For direct arm-only Galaxy XR control, use the separate G1 body-controller
launcher and the Thor launcher documented in
[`g1_xr_arm_teleop.md`](g1_xr_arm_teleop.md). In that mode, start
`scripts/run_g1_dreamwaq_body_controller.sh` on G1 and
`pixi run -e thor-teleop g1-arm-teleop` on Thor. Do not run
`run_g1_dreamwaq_controller.sh` or `g1_lowlevel_bridge` in that mode.

## G1 DDS and packet contracts

- `G1_PATH_UPPER_COMMAND` remains PATH's 192-byte PDU1 packet. It carries 14
  arm targets and grip flags; BOOST's adapter forwards the arm slots to
  `rt/arm_sdk`.
- `G1_POLICY_ACTION` GCA1 is a separate 29-joint policy packet. It is not a
  PATH-arm packet and is not consumed as one.
- `G1_ROBOT_STATE` GST1 remains 461 bytes and carries all 29 measured joint
  states plus base/torso IMU data. BOOST keeps its existing little-endian packet
  layout.
- `G1_CAMERA_FRAME` CRAM is the target g1_ws camera header. It remains a
  separate camera path; this integration's Teleimager wrist relay uses TIM2.

The schema reference is in [`../external/g1_ws/SOURCE.md`](../external/g1_ws/SOURCE.md).
Do not launch the source `g1_lowlevel_bridge` beside DreamWaQ: its smoke path
also initializes a `rt/lowcmd` publisher.

## Teleimager and wrist relay

The selected preset is
`external/teleimager/config/hbvcam_4m2214hd_g1.yaml`. It exposes head TIM2 on
55558, left wrist TIM2 on 55559, and right wrist TIM2 on 55560. The wrist
streams are 640×480. `scripts/run_g1_teleimager.sh` starts this one server and
does not start VideoHub or a second OpenCV camera process.

On Thor, source `configs/teleop/local.env` and launch
`services/pose_service/tools/run_thor_g1_teleimager_wrist_relay_v11.sh`. The
relay reads the configured Teleimager preset, subscribes to the TIM2 recording
ports, validates the 44-byte `<4sHHQQQIII>` header and image dimensions, then
publishes JPEG plus metadata as wrist RGB message version 2. The payload
includes camera sequence, G1 capture wall/monotonic timestamps, and Thor
receive timestamp. The worker still accepts version 1 messages during
transition and uses Thor receive time for freshness. It does not compare G1
and Thor monotonic clocks. Sequence gaps are counted and logged.

The existing D405 RGB-D gateway, depth synchronization, and 3D estimation
remain separate. The JPEG-only Teleimager wrist stream does not replace them.

## MuJoCo and Galaxy XR

Install the desktop environment and run the active table/objects scene:

```bash
export PIXI_FROZEN=true
pixi install -e sim
pixi run -e sim setup-sim
pixi run -e sim mujoco
```

Keep `PIXI_FROZEN=true` exported for subsequent desktop sim commands so Pixi
uses the committed sim lock without re-solving the separate Thor environment.

`pixi run -e sim mujoco-no-xr` starts the same sim and PATH bridge without the
headset. `pixi run -e sim mujoco-hand-test` adds the existing Revo2 MPPI runner
with a local JSON hand pose. `pixi run -e sim mujoco-hand-xr` enables Galaxy XR
hand input and the contact query. Use
`MUJOCO_SIM_CONFIG=configs/sim/mujoco_revo2_vacant.yaml` to select the vacant
scene.

In sim mode, `run_dds_sim.py` evaluates the vendored DreamWaQ policy against
MuJoCo state every target 20 ms control period and applies the target 29-joint
PD gains and the MuJoCo actuator limits. Arm targets come from PATH on
`rt/arm_sdk`; the Revo2 command route remains unchanged. The sim bridge does
not create an `rt/lowcmd` subscriber or hardware writer. Viewer keys W/S,
A/D, and Q/R request velocity. With the default NEW2-3 export, X does not
change the policy input; the toggle is available with exports that include an
explicit locomotion-mode contract.

The selected config and export are set in
`configs/sim/mujoco_revo2.yaml`. `BOOST_DREAMWAQ_EXPORT` can override the
checkpoint directory at launch. The table/object scene, Galaxy XR arm IK, and
Revo2 MPOPI route remain the integration scenario.
