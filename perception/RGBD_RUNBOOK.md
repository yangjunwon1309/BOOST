# Thor RGB-D validation runbook

## Production startup: G1 raw depth + Thor alignment

The process that already owns G1 low-level control must be the only low-level
bridge. Add these options to its existing command (do not start a second
controller just for state):

```bash
--publish-lcm-state \
--lcm-state-channel G1_ROBOT_STATE \
--lcm-state-rate 100
```

Run the independent VideoHub/depth bridge on G1:

```bash
cd /home/unitree/URL/29_dof_g1_deploy/g1_ws
export LCM_DEFAULT_URL='udpm://239.255.76.67:7667?ttl=1'

./build_rgbd_clean/g1_videohub_lcm_bridge eth0 \
  --depth-device /dev/video0 \
  --depth-width 640 \
  --depth-height 480 \
  --depth-fps 30 \
  --depth-scale-um 1000 \
  --rgb-width 640 \
  --rgb-height 360 \
  --alignment-mode thor-calibrated \
  --calibration config/d435_depth_to_videohub.yaml \
  --sync-tolerance-ms 20
```

Then start Thor:

```bash
cd /home/url/PATH-G1/perception

./tools/run_thor.sh python -m seg_web_node.web_server \
  --config /workspace/perception/config/pipeline.yaml \
  --lcm-url 'udpm://239.255.76.67:7667?ttl=1' \
  --calibration-channel G1_CAMERA_CALIBRATION \
  --robot-state-channel G1_ROBOT_STATE \
  --concepts 'object' \
  --enable-ee-publishing \
  --host 0.0.0.0 \
  --port 8080
```

Confirm the live geometry gates from Thor:

```bash
curl -s http://127.0.0.1:8080/api/status | jq '{
  calibration_id,
  lcm_calibration_frames,
  lcm_robot_state_frames,
  last_depth_source,
  alignment_ready,
  robot_state_delta_ms,
  torso_transform,
  geometry_runtime_error,
  ee_target
}'
```

Expected production values are `last_depth_source: 2`,
`alignment_ready: true`, a robot-state delta below 40 ms, a `dual_imu`
torso transform, and an empty EE blocker list after a valid segmentation.

## Offline validation (no G1 and no multicast)

```bash
cd /home/url/PATH-G1/perception

./tools/run_thor.sh python tools/validate_mock_rgbd.py \
  --image /workspace/perception/output/sam3_demo.jpg \
  --output /workspace/perception/output/mock_rgbd_overlay.jpg
```

This validates packet round-trip, timestamp pairing, zlib depth decoding,
vectorized deprojection, and overlay generation.

## Two-process LCM mock

Terminal 1:

```bash
cd /home/url/PATH-G1/perception
./tools/run_thor.sh python tools/rgbd_diagnostic.py \
  --count 10 \
  --timeout-s 30 \
  --camera-info /workspace/perception/config/camera_info.mock.yaml \
  --output-dir /workspace/perception/output/rgbd_diagnostic
```

Terminal 2:

```bash
cd /home/url/PATH-G1/perception
./tools/run_thor.sh python tools/mock_rgbd_publisher.py \
  --image /workspace/perception/output/sam3_demo.jpg \
  --width 640 \
  --height 360 \
  --fps 5 \
  --count 12
```

The launcher uses a unique container name per invocation so both commands can
run concurrently.

## Receive manual `resize-only` G1 output

```bash
./tools/run_thor.sh python tools/rgbd_diagnostic.py \
  --allow-unaligned \
  --count 30 \
  --timeout-s 30 \
  --output-dir /workspace/perception/output/g1_resize_only
```

Do not use `--allow-unaligned` for production grasp geometry.

## Receive calibrated G1 output

In production, color intrinsics arrive from the G1 on
`G1_CAMERA_CALIBRATION`; `camera_info.yaml` is now only a legacy fallback for
already-aligned recordings. Run without the bypass:

```bash
./tools/run_thor.sh python tools/rgbd_diagnostic.py \
  --count 30 \
  --timeout-s 30 \
  --output-dir /workspace/perception/output/g1_calibrated
```

If LCM warns about dropped large packets, raise UDP receive limits on Thor and
G1, then restart the processes:

```bash
sudo sysctl -w net.core.rmem_max=8388608
sudo sysctl -w net.core.rmem_default=8388608
```

For permanence, place those two values in a dedicated file under
`/etc/sysctl.d/` according to the board's administration policy.

## Live SAM3 web view

For the TensorRT one-time export/build and compatibility notes, see the
`TensorRT backend` section in `README.md`. Add `--backend tensorrt` and use a
single concept for minimum latency. Use `--backend pytorch` to fall back without
rebuilding or changing the RGB-D transport.

Start this receiver before starting the G1 bridge. During `resize-only` G1
bring-up, retain `--allow-unaligned`:

```bash
cd /home/url/PATH-G1/perception
./tools/run_thor.sh python -m seg_web_node.web_server \
  --config /workspace/perception/config/pipeline.yaml \
  --lcm-url 'udpm://239.255.76.67:7667?ttl=1' \
  --concepts 'object,person,robot part,metal frame,chair,box,tool' \
  --allow-unaligned \
  --host 0.0.0.0 \
  --port 8080
```

Open `http://<THOR_IP>:8080`. The page shows the SAM3 mask overlay, bounding
boxes, per-instance median aligned depth, a depth inset, pairing latency, model
latency, and receiver counters. The comma-separated concept list can be changed
from the page. SAM3 has no native automatic "segment everything" mode; coverage
is bounded by this vocabulary.

For calibrated `src=0` depth, remove `--allow-unaligned`.

## Segmented center -> continuous PATH EE target

The status API and object selector expose a robust 3-D center for each mask. It
is the component-wise median of valid aligned depth points inside the mask, not
the 2-D bounding-box center. Select an object and press **Track and send
continuously**. The server then republishes the latest center for the same
labeled instance, choosing the 3-D center nearest its previous position.

Publishing continues until **Stop** is pressed or the object is absent for
`tracking_lost_grace_frames` consecutive SAM3 frames. A temporary invalid depth
center pauses publishing without discarding the selection. A candidate that
jumps farther than `tracking_max_jump_m` is treated as lost instead of silently
switching to another object of the same class. The configured identity
quaternion remains fixed; segmentation supplies position only.

The server now evaluates
`T_pelvis_camera(t)=T_pelvis_torso(t) @ T_torso_camera` per RGB frame. The
default `dual_imu` estimator uses the pelvis and secondary torso IMUs for
relative attitude and the official G1 chain for the short link-origin
translation. `joint_fk` is available as a diagnostic fallback. `T_goal_cam`
remains supported only for old, waist-locked recordings.

To fit the custom mount transform from at least six well-spread, independently
measured corresponding 3-D target points:

```bash
./tools/run_thor.sh python tools/calibrate_torso_camera.py \
  /workspace/perception/config/torso_camera_correspondences.yaml \
  --output /workspace/perception/config/extrinsics.yaml \
  --max-rms-m 0.01
```

The input has two matching arrays named `camera_points_m` and
`torso_points_m`. Do not use an online matrix from another G1: the camera mount
is not in Unitree's robot model and a few millimeters of mount translation or a
few degrees of rotation can exceed the requested 1–3 cm object error.

After calibrating those two YAML files, start the PATH planner and state-machine
bridge, then launch perception with the publisher explicitly enabled:

```bash
cd /home/url/PATH-G1/perception
./tools/run_thor.sh python -m seg_web_node.web_server \
  --config /workspace/perception/config/pipeline.yaml \
  --lcm-url 'udpm://239.255.76.67:7667?ttl=1' \
  --concepts 'object,person,robot part,metal frame,chair,box,tool' \
  --enable-ee-publishing \
  --host 0.0.0.0 \
  --port 8080
```

Use `--ee-dry-run` together with `--enable-ee-publishing` to validate the web
controls, coordinate transforms, freshness checks, and workspace rejection
without emitting any ZeroMQ target packets. Never use `--allow-unaligned` for a
real EE command; the server rejects source 1 even if that diagnostic option is
present.
