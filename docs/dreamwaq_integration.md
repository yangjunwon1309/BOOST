# DreamWaQ integration

## Runtime boundary

The active hardware path retains Galaxy XR, PATH upper-body IK, and Revo2 hand
control. PATH sends its 14 arm targets as PDU1 over LCM. The arm-only G1
adapter publishes `rt/arm_sdk`; DreamWaQ reads that topic and `rt/lowstate`,
then sends the merged body target through the sole hardware `rt/lowcmd` writer.
The G1 adapter does not own `rt/lowcmd`.

The selected deployment source is `url_g1_deploy` commit
`3d37aa43f88964fc4255857c840dd33dd08e6eda`. The selected model/export pair is
`new3_4_backpack_440000`: its export manifest records a 12-action lower-body
policy and a backpack payload domain of 2–5 kg. The MuJoCo model adds the
nominal 3.5 kg box at the manifest center `[-0.12464276, 0, 0.1]` m with
`[0.11, 0.18, 0.34]` m dimensions. The export and payload metadata remain
together in `external/dreamwaq_g1_29dof/checkpoints/new3_4_backpack_440000/exported/`.
The manifest and policy files are hash-recorded in that directory's
`SOURCE.md`. The original training checkpoint is not needed to run inference.

The deployment config and action compiler retain the target joint order,
waist behavior, gains, torque limits, command timing, deadline handling, and
arm-target smoothing. BOOST does not layer additional reference filtering or a
backpack compensation torque onto that runtime. The selected export manifest
and target deployment config govern those values.

## G1 hardware launch

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

The hardware launcher starts the PATH adapter with arm output enabled and
`--hand-backend none`, then invokes the target `deploy/run_dreamwaq.py` with
the selected export, standing mode, 15 ms action deadline, and three-timeout
diagnostic window. It intentionally requires the explicit
`BOOST_ALLOW_HARDWARE_CONTROL=1` environment setting. The target runtime still
requires its START then A enable procedure; X switches standing/walking. Do
not start a second body controller or `g1_ws/g1_lowlevel_bridge` alongside it.
The `g1_ws` bridge publishes `rt/lowcmd` even in its null-action smoke mode.

The target runner's default CPU partition reserves camera cores 0–2, DDS I/O
cores 3–5, and selects policy/control cores from the remaining CPUs. The
Teleimager launcher uses camera cores 0–2. Check these masks against the G1's
available cores before deployment; change the target runner's CPU options and
`G1_TELEIMAGER_CAMERA_CPUSET` together if the layout differs.

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
pixi install -e sim
pixi run -e sim setup-sim
pixi run -e sim mujoco
```

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
A/D, and Q/R request velocity; X toggles DreamWaQ standing/walking.

The selected config and export are set in
`configs/sim/mujoco_revo2.yaml`. `BOOST_DREAMWAQ_EXPORT` can override the
checkpoint directory at launch. The table/object scene, Galaxy XR arm IK, and
Revo2 MPOPI route remain the integration scenario.
