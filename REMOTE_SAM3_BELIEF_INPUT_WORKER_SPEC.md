# Remote SAM3 Belief-Input Worker Specification

## 1. Purpose

Run SAM3 inference on an RTX 4060 Laptop GPU while preserving the existing
PATH RGB-D geometry stack. The laptop must produce the complete, validated
input domain for Thor-side parallel TASC belief inference:

```text
SAM3 masks
  -> synchronized filtered RGB-D object clouds
  -> pelvis-identity-frame object geometry and stable 3-D track IDs
  -> persistent cylinder surfaces and grip/contact candidates
  -> PATH_REMOTE_PERCEPTION_RESULT
  -> Thor gateway: ParallelTascBelief + A/B authority gate + EE target ZMQ
```

The laptop is a perception worker only. It must never publish
`path_static_ee_target`, `path_shared_control_hand`, arm commands, or any
other robot-control message. Thor remains the sole owner of belief target
selection and all control-side ZMQ publication.

This is intentionally **not** an implementation of TASC belief on the laptop.
The output is the geometric input needed by Thor's existing two-hand
`ParallelTascBelief`. This preserves one authoritative robot-state/FK view and
avoids dual target publishers.

## 2. Existing Code Assessment

| Existing component | Reuse | Reason |
| --- | --- | --- |
| `remote_offload/remote_sam3_worker.py` | Reuse transport, SAM3 backends, RGB-D synchronization, calibration/state buffering, heartbeat, result/overlay publishers | It already subscribes to all four Wi-Fi LCM inputs and publishes the versioned result protocol. |
| `remote_offload/remote_yolo11_worker.py` | Reuse its RGB-D geometry/candidate path, not its YOLO backend | It already has the required point filter, 3-D geometry fallback, stable tracks, persistent cylinder tracker, and `GripCandidate` serialization. |
| `remote_offload/remote_web_server.py` | Keep as the laptop web UI base; extend diagnostics only | It already consumes the same remote RGB, depth, result and heartbeat channels and does not affect control. |
| `remote_offload/thor_gateway.py` | Keep TASC belief, A/B authority selection, and EE target publishing on Thor | It currently reconstructs `BeliefInputObject` from `center_goal_m` and `contact_candidates`, then runs `ParallelTascBelief`. |
| `seg_web_node/web_server.py` | Reference behavior only | It is the canonical local implementation of parallel belief and overlay behavior, but it should not be reused as a second control publisher on the laptop. |

## 3. Non-Goals

- Do not move `ParallelTascBelief`, reference dynamics, IPOPT, or ZMQ target
  publishing to the laptop.
- Do not publish direct EE goals from the laptop.
- Do not create a second robot-state source or infer missing robot state from
  image timestamps.
- Do not run the Thor `seg_web_node.web_server` alongside the new remote SAM3
  worker for the same Wi-Fi stream; that would duplicate inference and make
  diagnostics ambiguous.
- Do not serialize full point clouds over Wi-Fi. Candidate/surface summaries
  are sufficient for the Thor belief stage.

## 4. Transport Contract

Use the existing protocol without changing LCM channel names.

| Direction | Channel | Owner | Payload |
| --- | --- | --- | --- |
| Thor -> laptop | `PATH_REMOTE_RGB_FRAME` | Thor gateway | Existing `CameraFrame` bytes |
| Thor -> laptop | `PATH_REMOTE_DEPTH_FRAME` | Thor gateway | Existing depth packet bytes |
| Thor -> laptop | `PATH_REMOTE_CAMERA_CALIBRATION` | Thor gateway | Existing calibration bytes |
| Thor -> laptop | `PATH_REMOTE_ROBOT_STATE` | Thor gateway | Timestamped `G1_ROBOT_STATE` packet |
| laptop -> Thor | `PATH_REMOTE_PERCEPTION_RESULT` | Remote SAM3 worker | Versioned MessagePack result |
| laptop -> Thor | `PATH_REMOTE_PERCEPTION_HEARTBEAT` | Remote SAM3 worker | Health/status MessagePack |
| laptop -> Thor | `PATH_REMOTE_OVERLAY_JPEG` | Remote SAM3 worker | Optional worker overlay JPEG |

The laptop uses only `WIFI_LCM_URL`, for example
`udpm://239.255.77.67:7667?ttl=1`. It must not subscribe to or publish on the
G1 Ethernet multicast group.

## 5. New Worker Structure

### 5.1 Module

Add `perception/remote_offload/remote_sam3_belief_input_worker.py`.

Implement `RemoteSam3BeliefInputWorker`, using `RemoteSam3Worker` as the
base for lifecycle, LCM, SAM3 backend selection, depth alignment, calibration,
state buffering, inference scheduling, heartbeat, and optional overlay
publication.

Do not copy the YOLO inference backend. Factor the reusable geometry code from
`RemoteYolo11Worker` into a small backend-agnostic helper module, for example
`remote_offload/remote_geometry_pipeline.py`, used by both workers:

```text
RemoteRgbdGeometryPipeline
  ObjectPointFilterConfig
  Stable3DTracker
  LastValidGeometryFallback
  PersistentCylinderSurfaceTracker
  build_object_geometry(...)
```

The helper accepts masks/boxes/scores/labels and returns protocol-ready object
maps. It has no LCM, web, SAM3, or YOLO dependency.

### 5.2 Required Worker Configuration

The new worker CLI mirrors `remote_sam3_worker.py` and adds:

```text
--config PATH to pipeline.yaml                         required
--enable-contact-candidates                            default true
--disable-contact-candidates
--object-max-points INT                                default tasc_belief.object_max_points
--geometry-fallback-ttl-frames INT                     default segmentation.<backend>.track geometry fallback TTL
--publish-overlay-jpeg                                 optional
```

Load these existing YAML sections without introducing a second numeric tuning
file:

```text
segmentation.sam3 / tensorrt settings
segmentation.depth_self_filter
cloud.object_filter
ee_target (depth limits, transform estimator, state match tolerance)
tasc_belief.grip
tasc_belief.pseudo
tasc_belief.object_max_points
```

For the RTX 4060 8 GB initial run, use SAM3 PyTorch FP16 only if the selected
SAM3 adapter supports it safely. Start at 2-3 Hz and 640x360 input transport;
the worker must drop superseded frames rather than queue GPU work. TensorRT
engines are GPU-specific, so a Thor-produced engine must not be reused on the
laptop.

## 6. Per-Frame Processing

The following order is mandatory and is shared with the existing remote YOLO
geometry path:

1. Synchronize RGB/depth by the original packet timestamps.
2. Align raw optical depth to RGB when required by the calibration packet.
3. Match `PATH_REMOTE_ROBOT_STATE` to the RGB timestamp using
   `robot_state_max_delta_ms`; do not use the newest unrelated state.
4. Compute `T_goal_camera` via the configured torso estimator and the matched
   state. The result frame is `pelvis_identity_fk`.
5. Give the SAM3 backend the image/depth/FK self-filter context where the
   backend supports it. The depth-based robot filter remains the sole source
   of robot pixel removal.
6. Run SAM3 for the configured concepts and remove duplicate masks exactly as
   the current SAM3 worker does.
7. For each current-frame mask call `extract_filtered_object_cloud` with the
   shared `ObjectPointFilterConfig`; preserve its erosion, depth edge, median/
   MAD, SOR/radius outlier, voxel, and filtered-depth behavior.
8. Deproject only filtered valid pixels; transform the bounded cloud into
   `pelvis_identity_fk` with timestamp-matched `T_goal_camera`.
9. Compute `center_camera_m`, `center_goal_m`, `center_pixel_uv`,
   `median_depth_m`, `valid_depth_points`, and point-filter telemetry from the
   same cloud.
10. Assign stable 3-D IDs by same-label nearest-neighbor association.
11. Update `PersistentCylinderSurfaceTracker` per track. If a valid cylinder
    exists, sample its grip candidates; otherwise call
    `generate_grip_candidates(points_goal, grip_config, up_goal=gravity_up)`.
12. Run `extract_pseudo_contacts` with the cylinder model and the same
    measured `up_goal`; serialize the one contact normal/quality used by the
    orientation posterior.
13. Apply `LastValidGeometryFallback` only according to its TTL contract.
    A fallback result must be marked explicitly and must retain its original
    source timestamp. It must not create a new fake track.
14. Serialize a result, publish it, then publish an optional overlay. Never
    block the next RGB-D pair on web rendering.

If calibration, intrinsics, timestamp-matched robot state, or `T_goal_camera`
is unavailable, still publish the SAM3 mask diagnostically, but set the 3-D
fields and candidate arrays to null/empty. Such an object is not a valid belief
input on Thor.

## 7. Result Schema Extension

Keep all required v1 `remote_offload.protocol.make_object` fields. Add only
backward-compatible optional fields; protocol validation must check their type
when present.

```json
{
  "track_id": "bottle:track0007",
  "label": "bottle",
  "score": 0.94,
  "center_goal_m": [0.38, 0.11, -0.05],
  "target_contact_goal_m": [0.37, 0.10, -0.04],
  "contact_candidates": [
    {
      "center_goal_m": [0.37, 0.10, -0.04],
      "approach_goal": [-0.8, 0.1, 0.5],
      "width_m": 0.06,
      "geom_score": 0.72,
      "source": "cylinder_surface"
    }
  ],
  "pseudo_contacts": [
    {
      "center_goal_m": [0.38, 0.11, -0.05],
      "normal_goal": [0.0, 0.0, 1.0],
      "fit_quality": 0.81,
      "source": "cylinder_axis",
      "snapped_to_gravity": true
    }
  ],
  "surface_model": {"type": "cylinder", "...": "existing schema"},
  "point_filter": {"...": "existing telemetry"},
  "geometry_fallback": false,
  "source_robot_state_delta_ms": 4.2,
  "goal_frame": "pelvis_identity_fk"
}
```

Add `pseudo_contacts`, `source_robot_state_delta_ms`, and `goal_frame` to
`protocol.validate_result`. Preserve unknown optional fields only when their
values are MessagePack-safe; never include NumPy arrays.

Extend `thor_gateway._objects_for_belief()` to reconstruct
`PseudoContactPoint` objects from `pseudo_contacts` and pass them through to
`BeliefInputObject`. Its existing `contact_candidates` reconstruction remains
unchanged. This lets the laptop own expensive geometry while Thor retains the
authoritative posterior update.

## 8. Thor Responsibilities After This Change

Thor gateway continues to:

1. validate source timestamps, calibration ID, image dimensions, monotonic
   sequence, result age, and result-to-forwarded-pair association;
2. discard invalid or stale geometry rather than extrapolating it;
3. derive both robot hand FK/velocity/orientation inputs from Thor's local
   robot-state stream;
4. run `ParallelTascBelief.build_shared_geometry(..., up_goal=...)` and
   `update_hands(...)` for left and right every belief tick;
5. use A/B only as per-hand authority gates;
6. publish the dual-hand `path_static_ee_target` schema to the current local
   endpoint for the state machine.

The shared geometry cache on Thor must use worker candidates/pseudo contacts
when present. It must not refit a cylinder or regenerate candidates from a
missing point cloud, because the point cloud stays on the laptop.

## 9. Remote Web View

`remote_offload/remote_web_server.py` is the correct base. It already reads
the remote RGB, depth, result, heartbeat, masks, point-filter telemetry,
surface model, and candidates over Wi-Fi LCM. It should remain a passive
observer.

The remote SAM3 worker does not calculate the final posterior: Thor's gateway
does. The laptop web view remains a passive RGB-D, segmentation, and geometry
observer. Final two-hand belief diagnostics are recorded directly on Thor,
without adding another Wi-Fi LCM stream.

### 9.1 Thor Belief CSV

`thor_gateway.py` records the latest bounded two-hand TASC diagnostic state to
an append-only CSV when started with `--belief-csv <path>`. The maximum record
rate is `--belief-log-hz` (default: 10 Hz). Each orientation candidate is one
row and the recorded state is:

```json
{
  "published_timestamp_us": 0,
  "source_rgb_timestamp_us": 0,
  "remote_sequence": 0,
  "selected_hand": "left|right|none",
  "hands": {
    "left": {
      "armed": false,
      "alpha": 0.0,
      "target": null,
      "objects": [],
      "orientation": {}
    },
    "right": {
      "armed": false,
      "alpha": 0.0,
      "target": null,
      "objects": [],
      "orientation": {}
    }
  },
  "result_valid": true,
  "result_age_ms": 0.0
}
```

Each row contains candidate position and surface normal; relation; normal-dot-
hand-axis, alignment error, alignment/hold/converge advantages, angular-speed
term, and weighted orientation advantage; candidate posterior; hand axis and
angular velocity; and the final orientation target. A/B selection is recorded
only as `armed`, `selected_hand`, and `alpha`; both posteriors continue to be
updated and logged when neither button is held.

### 9.2 Laptop UI Requirements

The remote web server renders the same RGB-D and geometry diagnostics as the
current local SAM3 web view:

- RGB video remains the latest LCM RGB frame rate, independently of SAM3 or
  belief update rate.
- The latest SAM3 masks, boxes, contact markers, surface-normal arrows, and
  cylinder diagnostics remain overlaid until the next inference result.
- Display selected contact point, normal arrow, result age, source state delta,
  and geometry fallback status from the remote worker.

The current `draw_belief_overlay` and HTML belief-table behavior in
`seg_web_node/web_server.py` are the rendering reference. Refactor shared
rendering helpers if practical, but the remote viewer must not import or start
the Thor web-server runtime.

Extend it with diagnostic-only display of:

- `source_robot_state_delta_ms` and result age;
- `goal_frame` and whether geometry is current/fallback/invalid;
- cylinder axis, radius, RMSE, angular coverage, stale frame count;
- selected candidate and all candidate scores;
- pseudo-contact normal, fit quality, and gravity snap state.
- final two-hand posterior/target analysis from Thor's belief CSV; no final
  posterior is reconstructed in the remote viewer.

Do not put a publish button or ZMQ socket in this remote web view. The existing
Thor web view may continue to show the final per-hand posterior/target state;
the laptop web view diagnoses only segmentation and geometry.

## 10. Docker and Runtime

Add a laptop-specific image, for example
`perception/docker/Dockerfile.remote-sam3`, rather than modifying the Titan XP
YOLO image. It needs:

- a CUDA/PyTorch version compatible with the laptop driver and RTX 4060;
- the current `requirements-remote-yolo11.txt` geometry/LCM/web dependencies;
- SAM3 PyTorch dependencies and the existing `sam3_backend.py` adapter;
- no Thor TensorRT, JetPack, or G1 DDS dependency.

Mount `perception/models` read-only for checkpoints and persistent HuggingFace/
Torch caches read-write. Run with `--network host` so Wi-Fi multicast and the
web UI work. The web viewer can run in the same image or a separate lightweight
container; both subscribe only and may coexist.

## 11. Initial CLI Contract

The final CLI should look like this; paths/images are names to be created by
the implementation, not existing commands yet.

```bash
# Laptop: SAM3 + complete geometry/candidate output, no robot command output.
cd /path/to/PATH-G1/perception
docker run --rm --gpus all --network host \
  -v "$PWD/models:/models:ro" \
  -v "$PWD/config:/workspace/perception/config:ro" \
  -v "$HOME/.cache/huggingface:/cache/huggingface" \
  path-g1-remote-sam3:rtx4060 \
  python3 -m remote_offload.remote_sam3_belief_input_worker \
    --wifi-lcm-url 'udpm://239.255.77.67:7667?ttl=1' \
    --config config/pipeline.yaml \
    --backend pytorch \
    --concepts bottle \
    --inference-hz 3 \
    --publish-overlay-jpeg

# Laptop: passive remote visualization.
docker run --rm --network host path-g1-remote-sam3:rtx4060 \
  python3 -m remote_offload.remote_web_server \
    --wifi-lcm-url 'udpm://239.255.77.67:7667?ttl=1' \
    --host 0.0.0.0 --port 8081

# Thor: existing Wi-Fi gateway, with belief enabled and the same YAML.
./tools/run_thor_remote_gateway.sh \
  --g1-lcm-url "$G1_LCM_URL" \
  --wifi-lcm-url "$WIFI_LCM_URL" \
  --config /workspace/perception/config/pipeline.yaml \
  --belief-inference
```

## 12. Acceptance Tests

### Unit

1. SAM3 worker uses the shared geometry helper and produces the same object
   geometry/candidate output as the YOLO worker for fixed masks/depth/state.
2. `pseudo_contacts` and cylinder fields pass protocol encode/decode/validate.
3. No robot-state match produces diagnostic masks but zero valid candidates.
4. A `15 deg` gravity tilt changes fallback candidate approach as expected.
5. Current and fallback geometry retain stable 3-D track IDs without duplicate
   old masks.
6. Remote web runtime renders result packets containing optional pseudo-contact
   fields without affecting the stream rate.
7. Thor belief CSV records both hand states even when neither A nor B is held.

### End-to-End

1. Thor forwards RGB, depth, calibration, and robot state; laptop heartbeat
   confirms all streams and reports matching source timestamps.
2. Laptop output has `center_goal_m`, candidate list, pseudo contact, and
   `source_robot_state_delta_ms` within the configured tolerance.
3. Thor gateway validates the packet and shows non-empty left/right TASC
   posteriors before A/B is held.
4. Holding A or B changes only the receiving hand's authority; neither action
   changes laptop inference or geometry publication.
5. Killing the laptop worker makes Thor mark the result stale and hold operator
   teleoperation; no stale remote candidate becomes a new target.
6. With no A/B button held, Thor continues to record both posteriors but does
   not inject either target into the control path.

## 13. Implementation Order

1. Extract shared geometry helper from `RemoteYolo11Worker` without behavior
   change; retain and test the YOLO worker.
2. Implement `RemoteSam3BeliefInputWorker` using that helper.
3. Add optional pseudo-contact fields to protocol and Thor reconstruction.
4. Extend remote web diagnostics.
5. Build the RTX 4060 SAM3 image and run mock transport validation.
6. Run SAM3 with a single label at 2-3 Hz; compare laptop geometry output with
   the existing Thor local SAM3 result on a recorded input.
7. Enable Thor belief inference only after timestamp, frame, candidate, and
   stale-result checks pass.
