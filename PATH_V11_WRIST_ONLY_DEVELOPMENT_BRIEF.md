# PATH v11 — Future Codex Development Brief: Two Wrist D405s Only

## Instruction to Codex

Implement this specification in the current `PATH-G1` repository. This is an
implementation request for a future run: inspect the repository first, make a
small coherent series of commits/changes, and verify them with focused tests.
Do not silently retain the head-camera path as the default. The production
perception and belief path described below uses **only two wrist-mounted Intel
RealSense D405 cameras directly attached to Thor**.

The previous head D435i transport and worker remain valuable for ablation.
Preserve that implementation under explicit `*_head_d435i_legacy` names using
`git mv`; do not delete it and do not make the new production path import it.

## Decisions already made

| Concern | Required v11 behavior |
|---|---|
| Camera source | Two D405 cameras, mapped by serial number to `left` and `right` wrists, capture on Thor. No head D435i frames participate in the v11 SAM3, geometry, or belief path. |
| Thor ingress | Read D405 color+depth directly through `pyrealsense2` on Thor. Do not receive camera frames from G1 LCM. |
| Thor → remote | Thor publishes wrist frame bundles and timestamped robot state to the remote SAM3 machine via the Wi-Fi LCM URL. |
| SAM3 input | Vertically stack the two color images: left on top, right on bottom. With two 640×360 images, SAM3 receives 640×720. Never horizontally concatenate. |
| Depth geometry | Split SAM3 masks back into their originating camera half before any geometry. Use each mask only with that D405's own native left-imager depth grid, intrinsics, timestamp, and wrist transform. |
| Belief | Object/contact belief uses the wrist observations and existing position/velocity-dominant advantage terms. It does not receive head-camera candidates. |
| Orientation source | Use live gravity as the implicit floor/surface normal for the floor relation. Do not require a noisy fitted D405 normal to create the floor orientation constraint. |
| Position sharing | Keep discrete/arg-max position target selection. Never interpolate positions between object candidates. |
| Orientation sharing | Use continuous, confidence-weighted orientation authority at the target level, upstream of the existing second-order orientation reference. |
| Legacy ablation | Old head-D435i gateway/worker/result-viewer code stays runnable under clear `head_d435i_legacy` names and separate launchers/channels. |

## Hardware facts and non-negotiable invariants

1. A D405 depth image is computed from its left and right stereo imagers. Its
   RGB/color stream comes from the **left depth imager**, not a separate RGB
   camera: Intel's D405 SDK path calls this "color stream comes from depth
   sensor," and its D405 datasheet labels it `Color (Left Imager)`. Thus RGB is
   a single view (not stereo RGB), while the depth and RGB use the same
   left-imager basis. No depth-to-separate-RGB extrinsic/alignment stage is
   required when the selected native profiles share the same grid.
2. On Thor, configure one RealSense pipeline per D405 serial; request matching
   native color/depth profiles and assert their same-grid relationship. Do not
   insert `rs.align(rs.stream.color)` into the v11 path. Store both raw device
   timestamps and host capture timestamps.
   
   The D405 exposes two color-formatted UVC paths: **UYVY is the direct Color
   Stream from the Left Imager** and supports 640×360×30; **YUY2/YUYV is the
   RGB+ISP output** and does not expose 640×360 on the observed hardware. V11
   must select the direct UYVY left-imager stream for its 640×360 geometry
   contract. Do not select a `/dev/video*` number in configuration; resolve a
   camera by D405 serial and validate its selected format/profile at startup.
   The RGB+ISP endpoint may be evaluated later as a separate, explicitly
   calibrated experiment; it is not the v11 default.
3. The two D405s are independent cameras. Vertical stacking is an efficient
   SAM3 batching representation, not a stereo rig. Do not triangulate across
   wrists, merge a seam-crossing mask, share intrinsics/extrinsics, or assume
   hardware synchronization between them.
4. Camera identity is mandatory end-to-end. Every frame, calibration, mask,
   object, contact candidate, visualization record, log row, and belief input
   must carry `camera_id` (`left_wrist` or `right_wrist`) and the source
   timestamp.
5. The direct D405 processes and the legacy head-D435i processes must have
   mutually exclusive launcher names and LCM channels. Starting v11 must not
   accidentally start the head path, and starting a legacy ablation must not
   compete for a D405 device.
6. Fail closed: no valid serial mapping, duplicate serial, missing color or
   depth profile, missing calibrated `T_wrist_camera`, stale robot state, or
   invalid native depth means no geometry/control output for that camera.

## Required repository migration before new code

Perform these moves with `git mv`, then repair imports, tests, module entry
points, documentation, and launch scripts that use them. Keep the moved files
behaviorally unchanged except for import/module-name repairs required by the
move.

| Current head-camera implementation | Required legacy name |
|---|---|
| `perception/remote_offload/thor_gateway.py` | `perception/remote_offload/thor_gateway_head_d435i_legacy.py` |
| `perception/remote_offload/remote_sam3_worker.py` | `perception/remote_offload/remote_sam3_worker_head_d435i_legacy.py` |
| `perception/remote_offload/remote_sam3_belief_input_worker.py` | `perception/remote_offload/remote_sam3_belief_input_worker_head_d435i_legacy.py` |
| `perception/remote_offload/remote_geometry_pipeline.py` | `perception/remote_offload/remote_geometry_pipeline_head_d435i_legacy.py` |
| `perception/remote_offload/thor_result_receiver.py` | `perception/remote_offload/thor_result_receiver_head_d435i_legacy.py` |
| `perception/remote_offload/remote_web_server.py` | `perception/remote_offload/remote_web_server_head_d435i_legacy.py` |
| `perception/tools/thor_remote_rgbd_gateway.py` | `perception/tools/thor_remote_rgbd_gateway_head_d435i_legacy.py` |
| `perception/tools/run_thor_remote_gateway.sh` | `perception/tools/run_thor_remote_gateway_head_d435i_legacy.sh` |
| `perception/tools/run_thor_remote_result_receiver.sh` | `perception/tools/run_thor_remote_result_receiver_head_d435i_legacy.sh` |

Keep generic low-level utilities that are not head-camera-specific only if
they are truly reusable. In particular, do not rename a shared protocol module
if it would disrupt an unrelated backend such as YOLO; create a separate v11
wrist protocol instead. Update any head-based YOLO module that imports the
moved legacy SAM3 base class so that it remains a runnable head-camera
ablation.

Add a small `perception/remote_offload/README` section or equivalent launch
table that makes the split unambiguous:

- `*_head_d435i_legacy`: old G1-LCM/head-D435i input; ablation only.
- `*_wrist_d405_v11`: direct Thor D405 input; production default.

Do not use compatibility aliases such as a new `thor_gateway.py` importing the
legacy gateway: that would make accidental head-camera use too easy. New v11
launch documentation must name the new module explicitly.

## New v11 module boundaries

Use names close to the following. Equivalent names are acceptable only when
they retain the explicit `wrist_d405_v11` identity.

| New responsibility | New module/file |
|---|---|
| D405 enumeration, serial validation, native left-imager color/depth capture, per-camera profile metadata | `perception/remote_offload/wrist_d405_capture_v11.py` |
| Thor-side direct capture, latest-frame policy, Wi-Fi LCM publishing, result validation, TASC belief ownership | `perception/remote_offload/thor_wrist_d405_gateway_v11.py` |
| Versioned wrist bundle/result wire schema and distinct channels | `perception/remote_offload/wrist_d405_protocol_v11.py` |
| One SAM3 inference on the vertical stack; seam rejection and mask attribution | `perception/remote_offload/remote_sam3_wrist_d405_worker_v11.py` |
| D405-specific per-camera depth-mask geometry and wrist-frame transforms | `perception/remote_offload/wrist_d405_geometry_pipeline_v11.py` |
| v11 Thor and remote launchers | `perception/tools/run_thor_wrist_d405_gateway_v11.sh`, `perception/tools/run_remote_sam3_wrist_d405_v11.sh` |
| V11 configuration | `perception/config/wrist_d405_v11.yaml` and `perception/config/wrist_d405_extrinsics_v11.yaml` |

Use a new LCM namespace, for example `PATH_V11_WRIST_*`; do not overload
`G1_CAMERA_FRAME`, `G1_DEPTH_FRAME`, or the old `PATH_REMOTE_*` camera
channels. Retain a version field in every MessagePack payload and reject a
wrong version or unknown `camera_id`.

## Thor runtime integration

The current generic Thor launcher mounts the source, models, and GPU but does
not expose USB/video devices. The v11 direct-capture launcher must therefore
be a dedicated wrapper, not an undocumented host-Python exception.

1. Add the tested `pyrealsense2` package version to the Thor image/runtime
   requirements. Pin the exact version only after it has passed the two-D405
   preflight on Thor; do not assume an import on the developer laptop proves
   the Jetson/Thor container can access a camera.
2. The v11 launcher must expose only the necessary USB/video devices and udev
   permissions to its container. Prefer explicit device mounts/cgroup rules
   for the two cameras over `--privileged`, and document the required Thor
   user/group/udev setup.
3. Add a container preflight command that imports `pyrealsense2`, lists the
   two configured serials and their profiles, then exits without starting
   streams. The existing `check_d405_stereo_basis.py` may be used as a
   topology diagnostic but is not a capture-health test.
4. The new direct-capture launcher must be the only v11 process allowed to
   open the D405 video nodes. It must report a clear conflict if a process
   such as `teleimager-serv` already owns them; it must not kill that process.

## V11 configuration contract

Create a checked-in, documented configuration template. It must require the
operator to fill in unique serials and calibrated wrist extrinsics rather than
guessing video-device indices.

```yaml
cameras:
  left_wrist:
    serial: REQUIRED_LEFT_D405_SERIAL
    color: {format: UYVY, source: left_imager, width: 640, height: 360, fps: 30}
    depth: {width: 640, height: 360, fps: 30}
  right_wrist:
    serial: REQUIRED_RIGHT_D405_SERIAL
    color: {format: UYVY, source: left_imager, width: 640, height: 360, fps: 30}
    depth: {width: 640, height: 360, fps: 30}

capture:
  require_native_left_imager_rgbd: true
  max_rgb_depth_delta_ms: 20
  stale_frame_ms: 150
  # One SAM3 input uses the newest valid pair from each wrist. This is not a
  # cross-camera hardware-sync guarantee; report the inter-wrist skew.
  max_inter_wrist_skew_ms: 80

geometry:
  min_depth_m: 0.07
  max_depth_m: 0.50
  min_depth_points: 100
  robot_state_max_delta_ms: 15
  seam_guard_px: 8
  allow_cross_wrist_candidate_merge: false

orientation:
  floor_normal_source: live_gravity
  # `gravity_up_in_pelvis` is the canonical source. A fitted local normal may
  # be telemetry only; it must not replace the floor constraint by default.
  use_fitted_surface_normals_for_floor_constraint: false
```

`wrist_d405_extrinsics_v11.yaml` must contain a distinct finite 4×4
`T_left_wrist_camera` and `T_right_wrist_camera`, with an unambiguous frame
convention: `p_wrist = T_wrist_camera @ p_camera`. Do not reuse the head
`T_torso_cam` or static `T_goal_cam` configuration. At each capture timestamp,
calculate

$$
T_{goal\leftarrow cam_i}(t)=T_{goal\leftarrow wrist_i}(q(t))\,T_{wrist_i\leftarrow cam_i}.
$$

The robot state selected for this transform must be the nearest valid state to
that camera's capture timestamp. If its age exceeds 15 ms, publish the image
diagnostic but omit goal-frame geometry and belief candidates for that camera.

### Wrist-holder CAD and extrinsic acceptance

The supplied Dex1 holder CAD lives at
`/home/url/Downloads/ff287f8f700948b5a30e3f4331a46b51/g1-wrist/dex1-1` and
shows the D405 installed upright above the wrist. It is a source for the
initial mechanical pose only. Those STEP files are individual printed parts,
not an assembly that defines both the `left_wrist_yaw_link`/`right_wrist_yaw_link`
origin and the D405 optical origin. Therefore Codex must not invent a numeric
translation from the thumbnails or STEP global coordinates.

For the stated approximation, use the assembled ring center as the wrist-yaw
origin and the midpoint of the symmetric camera-mount screw pair as the camera
center. The STEP geometry gives `(0, -70.878, -11.106)` mm and a 25° mounting
face tilt. The resulting initial transform is recorded in
`config/wrist_d405_extrinsics_v11.yaml`; it is explicitly CAD-only with 25 mm
translation and 10° orientation uncertainty. Before enabling autonomous
goal-frame geometry, fit each final 4×4 transform using a measured hand-eye or
correspondence procedure and record its residual. Verify the right-wrist
installation is not mirrored relative to the assumed upright holder.

Before hard-coding a 640×360 RealSense profile, add a preflight mode that lists
the profiles supported by both selected serials and reports the selected
native left-imager color/depth profiles, matching-grid assertion, USB
descriptor, and physical port. It must refuse to start if exactly the two
configured D405 devices are not present.

For factory intrinsics without `pyrealsense2`, use the C++ librealsense utility
`rs-enumerate-devices -c`. The repository command is:

```bash
python3 perception/tools/query_d405_intrinsics.py --ssh url@thor --json --raw
```

It requires `rs-enumerate-devices` (normally supplied by a `librealsense2-utils`
package or a librealsense build) on Thor, not the Python binding. Archive its
output with the physical-port mapping and selected 640×360 UYVY profile.

## Thor direct-capture and Wi-Fi protocol

The new Thor gateway owns the D405 capture. It must not subscribe to
`G1_CAMERA_FRAME` or `G1_DEPTH_FRAME`.

For each `camera_id`, capture a bundle containing:

- camera id and D405 serial;
- monotonic host capture timestamp plus SDK color and depth timestamps;
- color JPEG or an explicitly documented alternative;
- depth from that camera's native left-imager grid, depth scale, dimensions,
  and intrinsics valid for that exact frame/profile;
- calibration/extrinsic revision identifier;
- capture diagnostics: color/depth delta, dropped frames, matching-grid
  failure, and device-reconnect counter.

At each remote-inference dispatch, send one atomic dual-camera bundle. It must
contain the latest valid bundle for both wrists, the inter-wrist skew, and the
robot-state packet stream or the matched state needed to resolve each wrist
pose. A camera may be individually absent; do not substitute one wrist image
into the other half. In normal production mode, no dual-wrist bundle means no
new belief output. A one-camera diagnostic mode may exist, but it must be an
explicit CLI/config flag and must never be the default.

The remote worker result must retain, per object/candidate:

- `camera_id`, serial, source RGB/depth timestamp, and calibration revision;
- local-camera mask size and local pixel coordinates (not stacked coordinates);
- `T_goal_cam`/robot-state age used for geometry, or an explicit unavailable
  status;
- all existing geometry quality and fallback fields, with fallbacks limited to
  the same `camera_id`.

Keep the existing policy that Thor is the only component authorized to publish
control-side targets. The remote worker publishes perception geometry only.

## SAM3 vertical-stack pipeline

1. Decode the two D405 color images at the selected common size. Apply only
   documented color conversion; do not rescale either half independently.
2. Create `stacked_bgr = np.vstack([left_bgr, right_bgr])`, shape 720×640.
   Record `split_y = 360` in the dispatch payload. SAM3's square resize then
   has an anisotropy of 1.125, compared with 3.556 for a 1280×360 horizontal
   stack.
3. Run SAM3 once on the stacked image.
4. For every predicted mask, compute pixel support in the top and bottom half.
   A mask is attributable only if it has a strict majority in one half **and**
   no foreground pixels within `seam_guard_px` of the seam. Discard masks that
   touch/straddle the seam; never split one such mask into two objects.
5. Convert attributable masks and boxes to the local 640×360 coordinates of
   their source camera. Clip only after seam validation. Maintain independent
   tracking namespaces such as `left_wrist:<label>:track0001` and
   `right_wrist:<label>:track0001`.
6. Pair every local RGB mask with only its source D405's native left-imager
   depth. Do not vertically stack depth and do not run a single camera
   calibration over a stacked depth image.
7. Return overlay diagnostics with a visible seam, per-mask camera id, local
   source timestamps, depth validity, state age, and discarded-seam count.

Write focused unit tests for stack shape/order, inverse local-coordinate
mapping, seam rejection (including masks that barely touch the ±8 px zone),
and the guarantee that a left mask never consumes right depth/intrinsics.

## D405 depth surface preprocessing

The D405's near-range depth is better suited to wrist contact geometry, but
accuracy does not justify deleting safety filters without evidence. Begin with
an explicit audit of the current head-oriented chain in
`remote_geometry_pipeline_head_d435i_legacy.py` and related helpers.

Implement these changes:

1. **Remove the obsolete remote raw-depth-to-color reprojection path from the
   v11 worker.** D405 color is sourced from the left depth imager, so matching
   native color/depth profiles share the left-imager basis. Assert/profile-log
   that contract rather than inserting an SDK alignment pass. Intrinsics remain
   required for deprojection; this removes only a cross-sensor alignment step.
2. **Remove the fixed `GEOMETRY_COLOR_SIZE = (640, 360)` resampling layer in
   v11.** With selected matching 640×360 native color/depth profiles, geometry
   intrinsics and mask grid are already the input grid. Implement identity-grid
   assertions rather than silently resizing.
3. **Keep robot self-filtering initially, but make it wrist-aware.** A wrist
   camera sees its own hand/arm frequently, so the legacy torso-camera
   self-filter cannot be reused unchanged. Render/query the appropriate arm
   links in each camera frame using its dynamic `T_goal_cam_i(t)`. It may only
   be removed after a recorded ablation proves no self-body leakage.
4. **Keep robust depth validity, range, erosion/boundary, and outlier gates
   initially.** D405 errors and mixed-depth pixels still occur at contact
   boundaries. Add stage timing and before/after point-count diagnostics.
5. **Ablate, do not assume.** Add configuration switches and recorded-scene
   tests for bilateral smoothing, depth-edge veto, radius outlier removal, and
   voxel downsampling. Remove or relax a stage only when it improves the v11
   timing target without regressing contact-point variance, mask/depth
   association, self-body rejection, or invalid-depth rejection.
6. Restore `min_depth_points` to 100 at the native 640×360 grid unless the
   D405 recorded-data evaluation establishes a better threshold.

Use gravity from timestamp-matched robot state (`gravity_up_in_pelvis` or its
canonical successor) as the floor normal. Convert it into the current goal
frame using the same timestamped transform. This gravity surface is an
orientation reference only; it is not a fake object/contact point and must not
be mixed into object-position candidate selection. Fitted local D405 normals
may remain diagnostics or optional non-floor relations after validation.

## Wrist-only belief requirements

- Build object/contact candidates solely from the two per-camera v11 geometry
  outputs. Do not use head D435i objects, head calibration, or head image
  timestamps as fallback.
- Retain the existing position and velocity evidence as the dominant advantage
  terms. Preserve existing finite-value, timestamp, workspace, and contact
  quality gates.
- Preserve discrete position arbitration: one selected object/contact target
  for each output. A convex blend of object positions is invalid and is
  prohibited.
- Never cross-merge candidates or geometry fallbacks between wrist cameras by
  default. If future evidence needs cross-wrist association, add a separately
  tested, disabled-by-default fusion module that works in the goal frame and
  requires calibrated dynamic transforms.
- When a camera drops out or its state transform is stale, expire only that
  camera's tracks/candidates according to an explicit bounded TTL and report
  the reason. Do not republish a stale contact as a new observation.

## Continuous orientation shared control

Replace the current binary `orientation_valid` consumer path in
`path_streaming/path_state_machine_bridge.py`. The payload must continue to
carry `orientation_confidence`, but a perception-side confidence threshold must
not decide whether shared orientation abruptly exists.

### 1. Add reusable reference-dynamics primitives

In `path_core/intention/reference_dynamics.py`, add and unit-test:

- `slerp_so3(source, target, weight)`: geodesic SO(3) interpolation using the
  existing `rotation_vector` and `rotation_from_vector` helpers;
- `OrientationAuthority`: EMA confidence smoothing, Schmitt trigger, and a
  rate-limited authority weight;
- change `OrientationTargetGate.accept(...) -> bool` to an authority method
  returning a continuous factor in `[0, 1]`.

Use these initial values as config defaults, not hard-coded unchangeable
constants:

```yaml
orientation_shared_control:
  enter_confidence: 0.45
  exit_confidence: 0.25
  ema_alpha: 0.15
  max_authority_rate_per_s: 1.2
  gate_max_angle_rad: 1.047
  gate_taper_start_fraction: 0.667
```

`OrientationAuthority.update(confidence, gate_ok, dt)` must:

1. convert non-finite confidence to zero and apply EMA;
2. use enter/exit hysteresis (`0.45`/`0.25`) to suppress threshold chatter;
3. map the engaged smoothed confidence continuously from exit threshold to
   one, otherwise request zero authority;
4. rate-limit both engagement and release to 1.2 per second by default.

`OrientationTargetGate.authority(target, reference, dt)` must return zero for
no target, invalid target, an angle past `max_angle_rad`, or an unacceptable
target-rate jump. Within the final third of the admissible angle cone, taper
continuously to zero rather than hard-cutting. Preserve and multiply the
existing continuity/dwell safety logic into the returned factor. Retain clear
diagnostic reasons and counters.

### 2. Integrate only at the orientation target

For each hand in `path_state_machine_bridge.py`:

```python
rotation_command = human_tf[side][:3, :3]
rotation_target = rotation_command
constrained = rotation_target_from_relation(...)
gate_weight = orientation_gate[side].authority(constrained, current_rotation, dt)
belief_weight = orientation_authority[side].update(
    belief_target.orientation_confidence,
    gate_ok=gate_weight > 0.0,
    dt=dt,
)
weight = belief_weight * gate_weight
rotation_target = slerp_so3(rotation_command, constrained, weight)
r_ref = orientation_reference[side].step(rotation_target, dt, measured=..., ...)
```

Only call `slerp_so3` for a valid finite `constrained` rotation. Schedule the
orientation filter's natural frequency and damping continuously between the
released and armed settings using the final authority weight. Do **not** blend
the second-order reference state or multiply an already-generated reference by
a posterior: that would reintroduce a gap-proportional velocity step. Position
control remains unchanged and unblended.

The final authority is

$$w_{final}=w_{belief}\,w_{gate},$$

where `0` means pure operator orientation, `1` means fully enforced relation,
and intermediate values are soft virtual-fixture authority. Add it, both
components, gate reason, and the selected orientation relation to bounded
telemetry/logging.

### 3. Payload compatibility

During migration, producers may still emit `orientation_valid`; consumers must
treat it as diagnostic compatibility only. A valid finite orientation axis,
relation, frame, and `orientation_confidence` are the inputs to the new
authority path. Missing/invalid data must deterministically yield zero
authority, never retain an old constraint. Update the gateway's outgoing
payload formation so it does not recreate a hard confidence threshold.

## Verification and acceptance criteria

Add tests before declaring the work complete. Hardware tests must be skippable
in CI but documented and runnable on Thor.

### Unit/integration tests

1. Two configured serials are required and map deterministically to left/right;
   duplicate, missing, or non-D405 serials fail before capture.
2. A captured D405 bundle preserves per-device RGB/depth timestamps,
   intrinsics, depth scale, and native same-grid profile assertion. The gateway
   never subscribes to old G1 image/depth channels.
3. A 640×360 left image above a 640×360 right image produces 640×720 in the
   stated order. Local masks recover their correct source coordinates.
4. Seam-touching/straddling masks are discarded. No left mask can use right
   depth/intrinsics or vice versa.
5. Each geometry output uses the matching timestamped wrist FK transform;
   state age over 15 ms suppresses geometry/control output.
6. Gravity converts correctly into goal frame and supplies the floor
   orientation candidate without adding a position candidate.
7. Legacy imports and legacy launch `--help` commands still work after the
   rename, while new v11 imports do not depend on head legacy modules.
8. `slerp_so3` returns source at zero, target at one, and a geodesic midpoint.
9. Posterior oscillation around the threshold does not cause binary
   orientation-command toggles; authority follows EMA/hysteresis/rate limits.
10. Gate angle sweep through the taper interval decreases authority
    continuously; excessive angle/jump yields zero authority.
11. The second-order orientation reference respects `max_rate_rps` for all
    tested authority transitions, including loss of confidence during dwell.
12. Position target selection stays discrete under changing orientation
    authority.

### Thor hardware acceptance

1. Run `check_d405_stereo_basis.py --ssh <thor> --require-two` and record both
   serials before configuring v11.
2. The v11 preflight sees exactly those serials, reports the selected streams,
   and verifies 640×360 native left-imager RGB-D capture from both cameras.
3. With the legacy head process stopped, v11 runs for 10 minutes without
   device reconnect crash, video-device contention, stale-transform control
   output, or cross-camera mask attribution.
4. Measure and log: per-camera capture FPS, RGB/depth delta, inter-wrist skew,
   native-profile/grid assertion failures, remote latency, seam discard rate,
   valid depth points, geometry timing, robot-state age, and authority weight.
5. On representative near-contact recordings, compare the audited v11 surface
   preprocessing switches. A removed stage is accepted only if it meets the
   latency target and does not worsen contact-point frame-to-frame standard
   deviation, invalid-depth handling, or self-body rejection.
6. Confirm a floor-relation orientation command varies continuously as its
   confidence rises/falls and remains rate-bounded. Confirm no position target
   is geometrically blended.
7. Run the explicitly named head-D435i legacy launchers afterward and verify
   their original one-camera ablation still launches on a machine that has its
   legacy input source.

## Out of scope for this v11 implementation

- Reintroducing a head D435i fallback into the production v11 path.
- Treating the two wrists as an uncalibrated stereo pair or fusing them across
  the vertical seam.
- Automatically calibrating `T_wrist_camera`; the implementation consumes
  measured transforms and refuses geometry when they are absent.
- Blending object/contact positions.
- Removing geometry safety filters simply because D405 depth is more accurate.
- Changing low-level robot safety limits, arm control ownership, or allowing
  the remote worker to command the robot.

## Deliverables for the future implementation run

1. Renamed, runnable head-D435i legacy ablation path and its documentation.
2. Direct-Thor dual-D405 capture/gateway, new LCM schema, and remote SAM3
   vertical-stack worker.
3. Per-camera D405 depth-mask geometry using dynamic wrist FK and gravity-based
   floor orientation input.
4. Config templates, preflight/diagnostic CLI, and explicit v11 launchers.
5. Continuous orientation-authority primitives and bridge integration.
6. Focused tests plus a concise runbook with expected diagnostics and a record
   of the D405 profile/calibration assumptions.
