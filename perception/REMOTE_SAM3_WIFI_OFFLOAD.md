# PATH-G1 Remote Vision Wi-Fi Offload (head-D435i legacy ablation)

This document describes the preserved head-camera ablation path. Production
two-wrist D405 deployment is documented in `remote_offload/README.md`.

This mode moves vision inference and the visualization webview from Thor to an
external x86 desktop with a Titan XP. Thor stays responsible for G1 Ethernet
LCM, SONIC policy/control, result validation, and dry-run diagnostics. The first
integration stage never publishes EE targets or robot commands from Wi-Fi.

## Network Contract

Keep the two LCM domains separate:

```bash
export G1_LCM_URL='udpm://239.255.76.67:7667?ttl=1'
export WIFI_LCM_URL='udpm://239.255.77.67:7667?ttl=1'
```

Example interface names:

```bash
export G1_INTERFACE=enP2p1s0
export WIFI_INTERFACE=wlan0
```

Validate Thor routes before launch:

```bash
ip -brief addr show "$G1_INTERFACE"
ip -brief addr show "$WIFI_INTERFACE"
ip route get 239.255.76.67
ip route get 239.255.77.67
```

The G1 multicast route must resolve to the G1 Ethernet interface. The Wi-Fi
remote group must resolve to the Wi-Fi interface. Do not bridge Ethernet and
Wi-Fi:

```bash
bridge link
ip link show type bridge
```

On both Thor and the desktop, confirm they are on the same WLAN:

```bash
ip -brief addr
ping -c 3 <other-machine-wifi-ip>
```

If the AP blocks multicast or client-to-client traffic, use a WLAN/AP that
allows UDP multicast between clients, or introduce a unicast relay that forwards
the same channel payloads. Do not move remote perception traffic onto the G1
LCM multicast group.

## External Desktop Environment

Clone this repository on the Titan XP x86 desktop, then inspect the machine:

```bash
cd /path/to/PATH-G1
nvidia-smi
python3 --version
docker --version
ip -brief addr
```

Titan XP is Pascal/x86 (`sm_61`). Build a separate desktop environment; do not
reuse the Thor ARM64 Docker image and do not copy Thor TensorRT engines.
TensorRT plan files are tied to the GPU, TensorRT version, CUDA stack, and build
settings. The first Titan XP target is YOLO11 through PyTorch/Ultralytics, not
TensorRT.

PyTorch publishes CUDA wheel install commands from its official selector, and
Ultralytics supports Python `YOLO(...).predict(...)` with `imgsz`, `conf`,
`iou`, `classes`, `device`, and `quantize` arguments. This repo pins a known
CUDA 12.1 PyTorch wheel line in the remote YOLO11 Dockerfile and installs
Ultralytics without dependency resolution so it cannot replace that GPU stack.

Docker build on the external desktop:

```bash
cd /path/to/PATH-G1/perception
docker build -t path-g1-remote-yolo11:titan-xp \
  -f docker/Dockerfile.remote-yolo11 .
docker run --rm --gpus all --network host \
  -v "$PWD/models:/models" \
  -v "$PWD/.cache/torch:/cache/torch" \
  -v "$PWD/.cache/ultralytics:/cache/ultralytics" \
  path-g1-remote-yolo11:titan-xp
```

Cache the public YOLO11n segmentation checkpoint once:

```bash
docker run --rm --gpus all --network host \
  -v "$PWD/models:/models" \
  -v "$PWD/.cache/torch:/cache/torch" \
  -v "$PWD/.cache/ultralytics:/cache/ultralytics" \
  path-g1-remote-yolo11:titan-xp \
  python3 tools/cache_yolo11n_seg.py --output /models/yolo11/yolo11n-seg.pt
```

One Python venv option:

```bash
cd /path/to/PATH-G1/perception
python3 -m venv .venv-remote-yolo11
source .venv-remote-yolo11/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.4.0 torchvision==0.19.0 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r docker/requirements-remote-yolo11.txt
python -m pip install --no-deps ultralytics==8.4.108
```

SAM3 is intentionally not installed in a host venv. Use the dedicated RTX 4060
container described in `REMOTE_SAM3_BELIEF_INPUT_WORKER.md`, and keep Hugging
Face tokens outside repo files.

## Thor Gateway

Run this on Thor, separate from the existing SAM3 web server:

```bash
cd /home/url/PATH-G1/perception
python tools/thor_remote_rgbd_gateway.py \
  --g1-lcm-url "$G1_LCM_URL" \
  --wifi-lcm-url "$WIFI_LCM_URL" \
  --forward-hz 15 \
  --result-max-age-ms 800
```

Equivalent module form from the `perception` directory:

```bash
python -m remote_offload.thor_gateway_head_d435i_legacy \
  --g1-lcm-url "$G1_LCM_URL" \
  --wifi-lcm-url "$WIFI_LCM_URL" \
  --forward-hz 15 \
  --result-max-age-ms 800
```

The gateway subscribes to local G1 channels, timestamp-pairs RGB/depth, forwards
only the latest matched pair at the configured rate, periodically republishes
latched calibration and downsampled robot state, and validates incoming remote
results. It logs valid/invalid results only; it does not publish targets.

## Desktop YOLO11 Worker

Run this on the external desktop:

```bash
cd /path/to/PATH-G1/perception
docker run --rm --gpus all --network host \
  -v "$PWD/models:/models" \
  -v "$PWD/config:/workspace/perception/config:ro" \
  -v "$PWD/.cache/torch:/cache/torch" \
  -v "$PWD/.cache/ultralytics:/cache/ultralytics" \
  path-g1-remote-yolo11:titan-xp \
  python3 -m remote_offload.remote_yolo11_worker \
    --wifi-lcm-url "$WIFI_LCM_URL" \
    --config config/pipeline.yaml \
    --checkpoint /models/yolo11/yolo11n-seg.pt \
    --concepts 'object' \
    --inference-hz 3 \
    --extrinsics config/extrinsics.yaml \
    --publish-overlay-jpeg
```

`--concepts` maps to COCO labels through
`seg_web_node.yolo11_backend.Yolo11SegBackend`. Use `object`, `objects`, `all`,
or `*` for all allowed COCO classes except exclusions. Use comma-separated COCO
labels or aliases such as `bottle,cup,chair` to restrict GPU work and candidate
output. The worker publishes object masks, stable 3-D IDs, centers, and optional
geometry contact candidates in the same MessagePack result channel as SAM3.

For transport-only validation without YOLO weights:

```bash
python -m remote_offload.remote_yolo11_worker \
  --wifi-lcm-url "$WIFI_LCM_URL" \
  --backend mock \
  --concepts 'object' \
  --inference-hz 3
```

If YOLO/model state is unavailable, the worker publishes a fresh `valid=false`
result with an error string instead of reusing old coordinates.

## Desktop SAM3 Worker

### Pixi-managed host commands

The laptop host Python tooling is locked by the repository-root `pixi.toml`
and `pixi.lock`. This environment contains only CPU-side validation and launch
tools. CUDA, PyTorch, SAM3, TensorRT, and NVIDIA libraries remain inside the
Docker image, so installing or updating the Pixi environment cannot change the
host GPU driver or CUDA installation.

Install Pixi once, then install the exact locked environment:

```bash
curl -fsSL https://pixi.sh/install.sh | sh
export PATH="$HOME/.pixi/bin:$PATH"
cd /path/to/PATH-G1
pixi install --locked
pixi run --locked verify-host
pixi run --locked test-v6
```

Run the RTX 4060 TensorRT worker from terminal 1. The task uses host networking,
mounts the current source tree and the engine at
`perception/models/sam3_trt/rtx4060/sam3_fp16_4out.engine`, and automatically
falls back to `sudo docker` when the current user cannot access the Docker
socket:

```bash
cd /path/to/PATH-G1
export WIFI_LCM_URL='udpm://239.255.77.67:7667?ttl=1'
export SAM3_CONCEPTS='object'
export SAM3_INFERENCE_HZ='8'
pixi run --locked sam3-worker
```

Run the web viewer from terminal 2:

```bash
cd /path/to/PATH-G1
export WIFI_LCM_URL='udpm://239.255.77.67:7667?ttl=1'
export SAM3_WEB_PORT='8081'
pixi run --locked web-viewer
```

Open `http://127.0.0.1:8081`. To override a launch default, set one of
`SAM3_DOCKER_IMAGE`, `SAM3_TRT_ENGINE_HOST`, `SAM3_CONCEPTS`,
`SAM3_INFERENCE_HZ`, `SAM3_WEB_HOST`, or `SAM3_WEB_PORT` before `pixi run`.
The browser label input publishes `PATH_REMOTE_CONCEPT_CONTROL`, so labels can
be changed without restarting the worker.

Build and run the dedicated RTX 4060 container on the external laptop. Do not
install SAM3 into the host/root Python environment:

```bash
cd /path/to/PATH-G1/perception
docker build -t path-g1-remote-sam3:rtx4060 \
  -f docker/Dockerfile.remote-sam3 .
docker run --rm --gpus all --network host \
  -e HF_TOKEN \
  -v "$PWD/models:/models:ro" \
  -v "$PWD/config:/workspace/perception/config:ro" \
  -v "$HOME/.cache/huggingface:/cache/huggingface" \
  -v "$HOME/.cache/torch:/cache/torch" \
  path-g1-remote-sam3:rtx4060 \
  python -m remote_offload.remote_sam3_belief_input_worker_head_d435i_legacy \
    --wifi-lcm-url "$WIFI_LCM_URL" \
    --config config/pipeline.yaml \
    --backend pytorch \
    --concepts 'object' \
    --inference-hz 8
```

For transport-only validation without SAM3 weights:

```bash
docker run --rm --network host \
  -v "$PWD/config:/workspace/perception/config:ro" \
  path-g1-remote-sam3:rtx4060 \
  python -m remote_offload.remote_sam3_belief_input_worker_head_d435i_legacy \
    --wifi-lcm-url "$WIFI_LCM_URL" \
    --config config/pipeline.yaml \
    --backend mock \
    --concepts 'object' \
    --inference-hz 8
```

If SAM3/model state is unavailable, the worker publishes a fresh
`valid=false` result with an error string instead of reusing old coordinates.

## Desktop Webview

Run this on the external desktop:

```bash
cd /path/to/PATH-G1/perception
docker run --rm --network host \
  path-g1-remote-sam3:rtx4060 \
  python -m remote_offload.remote_web_server_head_d435i_legacy \
    --wifi-lcm-url "$WIFI_LCM_URL" \
    --host 0.0.0.0 \
    --port 8081
```

Open:

```text
http://<desktop-wifi-ip>:8081
```

The page shows RGB, segmentation overlay, aligned depth inset, per-object
stable ID, result timestamps, heartbeat data, latency, and link status. The
browser is visualization only; no machine-to-machine result is scraped from
MJPEG, HTML, or browser state.

## LCM Channels

Thor local input:

```text
G1_CAMERA_FRAME
G1_DEPTH_FRAME
G1_CAMERA_CALIBRATION
G1_ROBOT_STATE
```

Wi-Fi forwarded input:

```text
PATH_REMOTE_RGB_FRAME
PATH_REMOTE_DEPTH_FRAME
PATH_REMOTE_CAMERA_CALIBRATION
PATH_REMOTE_ROBOT_STATE
```

Wi-Fi desktop output:

```text
PATH_REMOTE_PERCEPTION_RESULT
PATH_REMOTE_PERCEPTION_HEARTBEAT
PATH_REMOTE_OVERLAY_JPEG
```

`PATH_REMOTE_PERCEPTION_RESULT` is MessagePack, version 1:

```text
{
  version: 1,
  sequence: int,
  published_timestamp_us: int,
  source_rgb_timestamp_us: int,
  source_depth_timestamp_us: int,
  calibration_id: int | null,
  image_width: int,
  image_height: int,
  backend: "sam3_pytorch" | "sam3_tensorrt" | "yolo11_pytorch" | "mock",
  inference_ms: float,
  valid: bool,
  error: string | null,
  t_goal_cam: [float x 16] | omitted,
  intrinsics: {fx, fy, cx, cy, width, height} | omitted,
  geometry_status: {
    state_available: bool,
    t_goal_cam_available: bool,
    up_goal_available: bool,
    state_delta_ms: float | null,
    primitive_count: int,
    fallback_count: int
  },
  objects: [
    {
      track_id: string,
      label: string,
      score: float,
      box_xyxy: [float, float, float, float],
      mask_encoding: "bitpack-zlib-row-major",
      mask_data: bytes,
      mask_height: int,
      mask_width: int,
      median_depth_m: float | null,
      center_pixel_uv: [int, int] | null,
      center_camera_m: [float, float, float] | null,
      center_goal_m: [float, float, float] | null,
      valid_depth_points: int,
      target_contact_goal_m: [float, float, float] | null,  # optional
      contact_candidates: [                                 # optional
        {
          center_goal_m: [float, float, float],
          approach_goal: [float, float, float],
          width_m: float,
          geom_score: float,
          source: string
        }
      ],
      surface_primitives: [                                 # optional/advisory
        {
          kind: "cylinder" | "plane" | "circle" | "pca" | "point",
          origin_goal_m: [float, float, float],
          axis_goal: [float, float, float],
          fit_quality: float,
          source: string,
          snapped_to_gravity: bool,
          radius: float,
          half_height: float,
          extent: [float, float, float] | null
        }
      ]
    }
  ],
  belief: optional reserved map
}
```

Masks are packed with `np.packbits(..., bitorder="little")`, compressed with
zlib, and must decode to the source RGB resolution. Pickle is never used.
Thor treats masks, identity, timestamps, calibration, boxes, scores, and
centroids as fatal contract fields. Malformed `surface_primitives`,
`contact_candidates`, `surface_model`, or `point_filter` entries are removed
individually and recorded in `soft_result_warnings` without discarding the
otherwise valid segmentation frame.

## Diagnostics

Mock end-to-end:

```bash
# Terminal 1, Thor
python tools/thor_remote_rgbd_gateway.py \
  --g1-lcm-url "$G1_LCM_URL" \
  --wifi-lcm-url "$WIFI_LCM_URL"

# Terminal 2, desktop
docker run --rm --network host \
  -v "$PWD/config:/workspace/perception/config:ro" \
  path-g1-remote-sam3:rtx4060 \
  python -m remote_offload.remote_sam3_belief_input_worker_head_d435i_legacy \
    --wifi-lcm-url "$WIFI_LCM_URL" \
    --config config/pipeline.yaml \
    --backend mock

# Terminal 3, desktop
docker run --rm --network host \
  path-g1-remote-sam3:rtx4060 \
  python -m remote_offload.remote_web_server_head_d435i_legacy \
    --wifi-lcm-url "$WIFI_LCM_URL" \
    --host 0.0.0.0 \
    --port 8081

# Terminal 4, Thor, using the existing mock publisher
python tools/mock_rgbd_publisher.py \
  --lcm-url "$G1_LCM_URL" \
  --image output/sam3_demo.jpg \
  --fps 5
```

Expected Thor gateway logs:

```text
forwarded_pairs increases
heartbeats increases
valid_results increases
invalid_results remains zero for the mock backend
```

Expected matching checks:

```text
source_rgb_timestamp_us equals the forwarded RGB timestamp
source_depth_timestamp_us equals the forwarded depth timestamp
calibration_id matches the latest latched calibration when available
mask dimensions equal image_width/image_height
stable IDs remain stable on repeated static frames
```

Performance counters:

```text
gateway: forwarded_pairs, dropped_before_forward, valid_results, invalid_results
worker: input_frame_count, processed_count, dropped_count, inference_latency_ms
webview: rgb_frames, depth_frames, results, result_age_ms
```

Safety checks:

```bash
# Disconnect Wi-Fi and confirm SONIC/control processes continue unaffected.
# Confirm Thor logs stale/missing remote results but emits no EE target packets.
# Confirm no process is launched that starts, stops, rebuilds, or modifies SONIC.
```
