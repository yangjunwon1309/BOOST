# Remote perception paths

`*_head_d435i_legacy` modules and the `run_thor_remote_*` launchers preserve
the former G1-LCM/head-D435i pipeline for ablation only.

Production wrist perception is explicit and separate:

```bash
# On Thor after rs-enumerate preflight identified the four UVC nodes and two USB paths.
# D405s are composite UVC devices: librealsense enumerates the complete UVC
# group, not only the Z16/UYVY nodes. The recorded Thor mapping has the two
# D405 groups at video0..5 and video6..11. Re-run topology/profile checks after
# reconnect; do not include unrelated video12 (the Sunplus webcam metadata).
export D405_VIDEO_DEVICES=/dev/video0,/dev/video1,/dev/video2,/dev/video3,/dev/video4,/dev/video5,/dev/video6,/dev/video7,/dev/video8,/dev/video9,/dev/video10,/dev/video11
export D405_USB_DEVICES=/dev/bus/usb/001/038,/dev/bus/usb/001/039
./tools/run_thor_wrist_d405_gateway_v11.sh --preflight
./tools/run_thor_wrist_d405_gateway_v11.sh \
  --g1-lcm-url "$G1_LCM_URL" --wifi-lcm-url "$WIFI_LCM_URL" \
  --belief-inference --belief-input-source operator_command \
  --belief-command-max-delta-ms 25 \
  --belief-command-divergence-max-m 0.20 \
  --ee-target-publishing \
  --ee-target-endpoint tcp://127.0.0.1:5564 \
  --ee-target-topic path_static_ee_target

# On the remote SAM3 host.
./tools/run_remote_sam3_wrist_d405_v11.sh --wifi-lcm-url "$WIFI_LCM_URL" \
  --backend tensorrt --trt-engine /models/sam3_trt/remote.engine
```

The v11 gateway opens D405s directly by serial and publishes only
`PATH_V11_WRIST_*` camera messages. It never subscribes to head RGB, depth, or
calibration channels. It receives wrist SAM3 geometry results and executes the
preserved head-gateway belief methods directly: 8 Hz shared geometry, 30 Hz
two-hand `ParallelTascBelief`, timestamp-matched operator commands with FK
fallback, selected-hand authority, and bounded telemetry on
`PATH_V11_WRIST_BELIEF_TELEMETRY`. The wrist viewer calls the preserved head
viewer belief-overlay routine on each camera using the newest telemetry; it
does not require image/result sequence matching. The gateway requires both
configured cameras by default and will not kill a conflicting V4L2 owner such
as `teleimager-serv`.

The selected belief target is published through the same legacy ZMQ contract
used by the head pipeline: `tcp://127.0.0.1:5564/path_static_ee_target`. The
unchanged PATH state-machine bridge applies `SecondOrderPositionReference` to
the selected position and the confidence-weighted `OrientationTargetGate`,
`OrientationAuthority`, and `SecondOrderOrientationReference` path to the
orientation. It then passes both absolute wrist transforms to the common
CasADi/IPOPT bimanual solver. A/B remains the dead-man selection for left/right;
without an active selection, belief telemetry is still visible but no control
target is emitted.

Before first capture, run `python3 tools/query_d405_intrinsics.py --ssh
url@thor --json --raw`, confirm left/right physical mounting with a covered
lens, and validate the CAD-only wrist extrinsics with a hand-eye procedure.
The production belief partition is strict: left belief receives only objects
segmented on `left_wrist`, and right belief only `right_wrist`; the global
gravity orientation reference remains available to both because it is not an
object/contact candidate. The telemetry fields `candidate_camera_id` and
`candidate_object_ids` expose that ownership.

For a measured replacement transform, record independently known pelvis-frame
fixture points together with D405 3-D points and either the corresponding
29-DoF robot state or `transform_diagnostics.t_goal_wrist`, then run:

```bash
cd ~/PATH-G1/perception
python3 tools/calibrate_wrist_d405_extrinsics_v11.py observations.yaml \
  --side left_wrist \
  --output config/wrist_d405_extrinsics_v11_calibrated.yaml
```

It solves `p_wrist = T_wrist_camera p_d405_optical`, rejects gross matching
outliers, and refuses output above 10 mm RMS by default. The result packet
also carries `transform_diagnostics` with the current camera-to-pelvis basis
vectors, so a recorded frame can be inspected before control is enabled.
For a fixed calibration target that SAM3 segments, this helper appends the
timestamp-matched `center_camera_m` and `t_goal_wrist` directly from the v11
result stream; repeat it for each target view:

```bash
pixi run --locked python tools/record_wrist_d405_calibration_observation_v11.py \
  --lcm-url "$WIFI_LCM_URL" --side left_wrist --label calibration_ball \
  --goal-point-m 0.420 0.180 0.260 \
  --output observations_left.yaml
```

For a dot initially placed exactly 20 cm along `+X` from
`left_wrist_yaw_link`, use the wrist-local form **once** to print its pelvis
coordinate. Then leave the dot fixed in the environment, move the wrist, and
use the printed value with `--goal-point-m` for the remaining views. Repeating
a dot mounted on the same wrist cannot calibrate the full transform:

```bash
pixi run --locked python tools/record_wrist_d405_calibration_observation_v11.py \
  --lcm-url "$WIFI_LCM_URL" --side left_wrist --wrist-point-m 0.20 0 0 \
  --output observations_left.yaml
```

The checked-in configuration is the USB-2 reduced mode: native left-imager
UYVY + Z16 at 480×270@15 and an 8 Hz latest-frame dispatch. Restore
640×360@30 only after both cameras negotiate USB 3.x/SuperSpeed.

The worker rejects an entire bundle when its paired Thor robot-state receipt is
older than 15 ms, reports this in `geometry_status.state_age_rejects`, and
publishes `source_bundle_sequence` with every result. The viewer retains a
bounded image/result/belief history and draws masks and belief targets only
when all three refer to that same result epoch. Its live raw image therefore
continues during SAM3 latency, but it deliberately has no stale overlay.
