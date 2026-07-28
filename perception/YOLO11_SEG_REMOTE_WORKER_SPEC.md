# YOLO11-seg Remote Perception Worker Spec

This note is for the external desktop implementation.  The desktop should not
reuse the Thor web view as the transport.  It should subscribe to Wi-Fi LCM RGB-D
inputs, run YOLO11-seg plus RGB-D/belief post-processing, and publish a compact
MessagePack result back to Thor.

## Scope

The current Thor implementation uses `seg_web_node.yolo11_backend.Yolo11SegBackend`
behind `python -m seg_web_node.web_server --backend yolo11-seg`.  For the remote
desktop worker, preserve the same segmentation and filtering behavior, but host
the diagnostic web UI on the desktop only.  Control-side publishing stays on
Thor after Thor receives and validates `PATH_REMOTE_PERCEPTION_RESULT`.

The desktop worker must never publish robot commands or EE targets directly.

## Input Transport

Thor forwards existing G1 packets over a dedicated Wi-Fi LCM group:

| Direction | Channel | Payload |
|---|---|---|
| Thor to desktop | `PATH_REMOTE_RGB_FRAME` | existing `CameraFrame` bytes |
| Thor to desktop | `PATH_REMOTE_DEPTH_FRAME` | existing depth bytes |
| Thor to desktop | `PATH_REMOTE_CAMERA_CALIBRATION` | existing calibration bytes |
| Thor to desktop | `PATH_REMOTE_ROBOT_STATE` | existing robot-state bytes |
| Desktop to Thor | `PATH_REMOTE_PERCEPTION_RESULT` | MessagePack result |
| Desktop to Thor | `PATH_REMOTE_PERCEPTION_HEARTBEAT` | small diagnostic heartbeat |

Use a separate Wi-Fi multicast URL from the G1 Ethernet LCM URL.  Do not bridge
the G1 Ethernet group onto Wi-Fi.

The desktop worker should synchronize RGB and depth by the original timestamps.
Use the same tolerance as Thor bring-up unless changed explicitly:

```text
sync_tolerance_ms = 50
```

If depth source is raw depth optical, use `PATH_REMOTE_CAMERA_CALIBRATION` to
align depth to RGB before computing 3-D centers and contact points.

## YOLO11-seg Backend

Default backend configuration currently used on Thor:

```yaml
yolo11:
  checkpoint: /models/yolo11/yolo11n-seg.pt
  imgsz: 640
  confidence: 0.80
  iou_threshold: 0.50
  max_masks: 20
  quantize: 16
  allowed_classes: []
  excluded_classes: [person]
  track:
    missing_ttl_frames: 6
    fallback_score_decay: 0.985
    match_iou: 0.15
    boundary_crop_px: 4
  robot_exclusion:
    arm_radius_px: 22
    wrist_radius_px: 42
    static_bottom_fraction: 0.16
    static_side_fraction: 0.035
    min_remaining_fraction: 0.30
```

Use Ultralytics YOLO11 segmentation with the COCO-80 pretrained
`yolo11n-seg.pt` checkpoint:

```python
model.predict(
    source=image_bgr,
    imgsz=640,
    conf=0.80,
    iou=0.50,
    max_det=20,
    classes=resolved_coco_class_ids,
    device=0,
    quantize=16,
    retina_masks=True,
    verbose=False,
)
```

`retina_masks=True` is intentional.  It is a little slower, but it preserves
mask quality at the original image size.  `quantize=16` means PyTorch FP16 in the
current Ultralytics API, not TensorRT.

## Concept Filter

The web prompt/concept field is not used as text grounding.  It is parsed as a
comma-separated list of exact COCO labels.

Wildcard concepts are:

```text
"", "*", "all", "object", "objects"
```

Wildcard means every COCO class except `excluded_classes`.  Otherwise each
concept must resolve to one COCO class.  Unknown labels produce no masks and
must not fall back to all-class inference.

Supported friendly aliases:

| User concept | COCO label |
|---|---|
| `television` | `tv` |
| `sofa` | `couch` |
| `table` | `dining table` |
| `plant` | `potted plant` |
| `phone`, `mobile phone` | `cell phone` |
| `ball` | `sports ball` |

`person` is excluded by default.  This is a manipulation safety guard and also
helps avoid operator/robot-adjacent masks.

## Post-processing Order

Apply post-processing in this exact order:

1. Resolve concepts to COCO class IDs.
2. Run YOLO11-seg with `conf >= 0.80`.
3. Resize masks to RGB resolution with nearest-neighbor if needed.
4. Drop excluded labels and labels outside `allowed_classes`.
5. Subtract robot pixels using FK-projected robot arm/hand mask.
6. Erode accepted masks inward by `boundary_crop_px = 4`.
7. Drop masks whose remaining area is below `min_area_px`.
8. Sort detections by detector score descending and keep at most `max_masks`.
9. Associate current detections to previous tracks by same label plus mask IoU.
10. Return only current-frame detections to RGB-D geometry and belief.

The edge crop is implemented as one elliptical erosion:

```python
diameter = boundary_crop_px * 2 + 1
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
```

This is important for depth.  YOLO mask boundaries often include background
pixels; using those pixels can move the median 3-D center or contact estimate.

## Tracking And Ghost Prevention

Track association is only a short memory for re-identification.  Missing tracks
are retained internally for up to `missing_ttl_frames = 6`, with score decay:

```text
fallback_score = previous_score * fallback_score_decay ** missing_frames
```

However, missing tracks must never be rendered, sent into RGB-D geometry, or fed
to belief.  They are internal candidates only.

The visible/output mask list is:

```python
[
    track for track in track_priors
    if track.missing == 0 and track.score >= confidence
]
```

This fixes the old "ghost mask" behavior during rapid viewpoint changes.  If
the detector sees the same label at a new image location, drop the unmatched old
same-label prior instead of publishing both old and new masks.  The old mask is
a stale viewpoint hypothesis, not a second object.

## Robot Arm And Hand Exclusion

YOLO COCO has no robot class, so robot self-masks must be filtered separately.
The Thor pipeline projects the current G1 arm/hand FK into the RGB image and
creates a binary exclusion mask.

Use the same radii:

```text
arm_radius_px = 22
wrist_radius_px = 42
```

For each YOLO mask:

```python
remaining = mask & ~robot_exclusion_mask
remaining_fraction = remaining.sum() / max(1, mask.sum())
```

If `remaining_fraction < min_remaining_fraction` (`0.30`), drop the whole mask.
Otherwise keep the trimmed mask.

The static fallback guards are:

```text
static_bottom_fraction = 0.16
static_side_fraction = 0.035
```

These are conservative image-border robot/body exclusion hints when dynamic FK
exclusion is not available, but dynamic FK exclusion should be preferred.

## RGB-D Geometry

After segmentation, deproject aligned depth using the color intrinsics:

```text
z_m = depth_raw * depth_scale_um / 1_000_000
x_m = (u - cx) * z_m / fx
y_m = (v - cy) * z_m / fy
```

Only accept depth samples in the configured valid depth range.  The current Thor
geometry uses the component-wise median of valid masked points:

```text
center_camera_m = median(xyz_camera[mask])
center_pixel_uv = median(valid mask pixels)
valid_depth_points = count(valid mask pixels)
```

Transform to the goal/pelvis frame with:

```text
center_goal_m = T_goal_camera @ center_camera_m
```

For belief/contact, also send a bounded per-object point cloud sampled from the
masked goal-frame points.  The Thor default cap is `tasc_pointcloud_max_points`.

Objects with insufficient valid depth may still be displayed diagnostically, but
they must not become valid belief/contact targets.

## Belief And Contact Result

The desktop worker should run the same TASC-style belief/contact logic after
segmentation and RGB-D geometry:

1. Assign stable object IDs using label plus 3-D nearest-neighbor tracking.
2. Generate contact/grip candidates from each object's masked goal-frame cloud.
3. Maintain contact candidate persistence using 3-D nearest-neighbor matching,
   not per-frame array index.
4. Compute object posterior and contact posterior.
5. Use object posterior as the action blending confidence; keep
   `joint_confidence = p(object) * p(contact | object)` only as diagnostic
   confidence.
6. Publish the selected contact point if the selected object/contact is valid;
   otherwise publish a valid null fallback result.

The selected target object should include:

```text
track_id
label
object_confidence
joint_confidence
selected_grip_index
target_contact_goal_m
center_goal_m
center_camera_m
center_pixel_uv
valid_depth_points
```

`target_confidence` should not be used as a field name because it is ambiguous.
Use `object_confidence` and `joint_confidence`.

## Remote Result Packet

The existing v1 protocol lives in `remote_offload/protocol.py`.  Keep its base
fields:

```text
version
sequence
published_timestamp_us
source_rgb_timestamp_us
source_depth_timestamp_us
calibration_id
image_width
image_height
backend
inference_ms
valid
error
objects
```

Each object currently requires:

```text
track_id
label
score
box_xyxy
mask_encoding
mask_data
mask_height
mask_width
median_depth_m
center_pixel_uv
center_camera_m
center_goal_m
valid_depth_points
```

For this YOLO11/belief remote worker, extend each object with optional belief
fields:

```text
object_confidence
joint_confidence
selected_grip_index
target_contact_goal_m
grip_candidates
```

Also add a top-level selected target field if the Thor consumer expects one:

```text
target: null | {
  track_id,
  label,
  object_confidence,
  joint_confidence,
  selected_grip_index,
  target_contact_goal_m,
  center_goal_m,
  source_rgb_timestamp_us
}
```

If no valid target exists, publish `target: null` and `valid: true` unless the
whole inference failed.  This allows the Thor-side state machine to hold its
default posture without treating a missing object as a transport error.

Masks should use the existing encoding:

```text
mask_encoding = "bitpack-zlib-row-major"
```

Do not put JPEG overlays on the control/result channel.  Overlays belong only in
the desktop diagnostic web UI.

## Timing Fields

The Thor web implementation now reports split timings:

```text
jpeg_decode_depth_align
yolo_model
rgbd_belief_overlay
web_jpeg_encode
end_to_end
```

The remote desktop worker should log the same breakdown, but only
`inference_ms` is required in the MessagePack result.  If performance is worse
than model-only benchmark, first inspect RGB-D alignment, belief/contact point
cloud processing, and JPEG/web rendering cost before changing the YOLO model.

## Validation Checklist

Before connecting to Thor control consumers, verify:

```text
1. Invalid concept, e.g. "tool" if not mapped to COCO, produces zero objects.
2. "bottle" only runs YOLO with the bottle class ID and emits no other labels.
3. All emitted object scores are >= 0.80.
4. Person detections are absent even in wildcard/object mode.
5. A rapid viewpoint change does not leave stale masks in the result packet.
6. Missing tracks may remain in internal metadata, but not in objects[].
7. Robot arm/hand projected pixels are removed before belief.
8. Every emitted 3-D target has non-null center_goal_m and enough valid depth.
9. Null target fallback is published when no valid object/contact exists.
10. Thor receiver accepts monotonic sequence, image size, calibration_id, and lag.
```

The relevant Thor-side reference files are:

```text
seg_web_node/yolo11_backend.py
seg_web_node/web_server.py
seg_web_node/geometry.py
seg_web_node/tasc_belief.py
remote_offload/protocol.py
remote_offload/thor_gateway.py
remote_offload/thor_result_receiver.py
config/pipeline.yaml
```
