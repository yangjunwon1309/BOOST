# G1 SAM3 perception environment (Jetson AGX Thor)

RGB-D transport, mock validation, and G1 interoperability commands are in
[`RGBD_RUNBOOK.md`](RGBD_RUNBOOK.md).
Remote x86/Titan XP Wi-Fi offload for YOLO11/SAM3 inference and the external
webview is documented in
[`REMOTE_SAM3_WIFI_OFFLOAD.md`](REMOTE_SAM3_WIFI_OFFLOAD.md).

This directory provides the isolated SAM3-only runtime specified by the G1
Perception Pipeline v2 document. It deliberately contains no AnyGrasp,
MinkowskiEngine, Open3D, PyTorch3D, or pycocotools dependency.

## Validated host target

- Jetson AGX Thor (`aarch64`, compute capability 11.0)
- JetPack 7.1 / L4T 38.4
- Ubuntu 24.04
- Host CUDA/driver 13.0 / 580.00
- Docker with the NVIDIA runtime

The base is `nvcr.io/nvidia/pytorch:26.06-py3`, not an `-igpu` image. The
previously installed `26.02-py3-igpu` image only carries `sm_87` kernels and is
not compatible with Thor's `sm_110` GPU.

The Dockerfile pins the multi-architecture NGC manifest digest so rebuilding
does not silently select a different framework image.

## Build and verify

```bash
cd /home/url/PATH-G1/perception
docker compose -f docker-compose.thor.yaml build verify
docker compose -f docker-compose.thor.yaml run --rm verify
```

Or use the wrapper:

```bash
./tools/run_thor.sh
```

The verifier runs a real CUDA matrix multiplication and reports the installed
SAM3 builder signatures. It does not download or instantiate the 848M-parameter
checkpoint.

For the pinned SAM3 revision, introspection on this image reports:

```text
build_sam3_image_model(..., checkpoint_path=None, load_from_HF=True, ...)
build_sam3_video_predictor(*model_args, gpus_to_use=None, **model_kwargs)
Sam3Processor(model, resolution=1008, device='cuda', confidence_threshold=0.5)
```

Keep these details inside the future `sam3_engine` adapter; application code
must continue to depend only on the specification's `SegInstance` interface.

## Checkpoints

SAM3 checkpoints are gated. Request access at:

https://huggingface.co/facebook/sam3

Do not put a Hugging Face token in the Dockerfile, Compose file, YAML config, or
Git. Authenticate into the persistent mounted cache:

```bash
./tools/run_thor.sh hf auth login
./tools/run_thor.sh python tools/cache_models.py
```

The second command caches both the official SAM3 image model and the default
`dinov2_vits14` re-identification model. To cache only one:

```bash
./tools/run_thor.sh python tools/cache_models.py --skip-dinov2
./tools/run_thor.sh python tools/cache_models.py --skip-sam3
```

Verify the cached DINOv2 patch-token path used by masked pooling:

```bash
./tools/run_thor.sh python tools/verify_dinov2.py
```

For an offline deployment, place manually obtained weights under `models/` and
set the corresponding `checkpoint` path in `config/pipeline.yaml`. The future
SAM3 adapter must inspect the installed builder signature before passing that
path; it must not assume an upstream keyword name.

## Runtime properties

The wrapper supplies:

- host networking for LCM multicast and the FastAPI port;
- NVIDIA GPU access;
- host IPC and inference-friendly `ulimit` values;
- persistent Hugging Face and Torch caches;
- persistent `models/` and `registry_data/` mounts;
- the host UID/GID, preventing root-owned output files.

The launcher also supplies `USER`/`LOGNAME` and persistent writable compiler
caches. This is required because the host UID may not exist in the NGC image's
`/etc/passwd`; without it, PyTorch Dynamo can fail in `getpass.getuser()` while
SAM3 imports torchvision.

Launch the SAM3 RGB-D web node with:

```bash
./tools/run_thor.sh python -m seg_web_node.web_server \
  --config config/pipeline.yaml \
  --prompt "object" \
  --port 8080
```

## EfficientViT-SAM-L0 generic-mask backend

`efficientvit-l0` is a text-free alternative for the robot-state/belief
pipeline. It returns generic `object` masks only. The RGB-D server then uses
the existing depth deprojection and 3-D nearest-neighbour tracker to assign
stable IDs such as `object:track0007`; it does not use SAM3 or YOLO labels.

The current Thor image (`g1-perception-sam3:thor-jp71`) includes the pinned
EfficientViT and Segment Anything sources. Cache its public checkpoint once:

```bash
./tools/run_thor.sh python tools/cache_efficientvit_l0.py
```

`reinitialization: central` is the default. It runs automatic generic-mask
discovery from a configurable 640×360 central ellipse (`center=(320,200)`,
`radius=(220,125)`, `spacing=50 px`) and then refines every discovered object
in one batched point/box-prompt decoder call. Each object keeps a local box
prior; the RGB-D server assigns the persistent 3-D `object:trackNNNN` ID.

Robot/self pixels are removed by the shared depth self-filter, not by
image-region or wrist heuristics. The server uses synchronized LCM depth,
camera intrinsics, camera extrinsics and `G1_ROBOT_STATE` to render padded FK
collision primitives into a robot depth buffer, then vetoes segmentation pixels
at or behind that depth with dual-margin hysteresis. `all` uses the full-image
grid and is useful only as an explicit broad scene reset because it creates
periodic GPU bursts.

```bash
./tools/run_thor.sh python -m seg_web_node.web_server \
  --config config/pipeline.yaml \
  --backend efficientvit-l0 \
  --allow-unaligned --host 0.0.0.0 --port 8080

# Standalone synthetic RGB-D validation; it does not start the web server.
./tools/run_thor.sh python tools/benchmark_efficientvit_l0_rgbd.py \
  --reinitialization central --iterations 120
```

On this Thor with the included `output/sam3_demo.jpg` and its synthetic mock
depth, central multi-object tracking retained four masks and measured 61.8 ms
mean (16.2 Hz), 253.8 ms p95 with a central reset every 15 frames. The normal
four-object tracking frame was about 44–47 ms; reset frames dominate the p95.
The full-image `all` path was 59.3 ms mean and 467 ms p95, so central is the
safer discovery mode even though it still needs a lower reset rate when SONIC
shares the GPU.

## TensorRT backend

The linked `dataplayer12/SAM3-TensorRT` project is a community implementation,
not an official Meta SAM3 deployment repository. Its Thor benchmark uses
TensorRT 10.13.3, CUDA 13.0, MAXN, and a C++ zero-copy path. This deployment
adapts its full-model ONNX approach to the existing Python RGB-D web pipeline.

The Docker image contains TensorRT and NVIDIA ModelOpt. Build the engine on the
Thor where it will run; TensorRT plans are specific to the GPU and TensorRT
version. The first two commands are one-time operations and reuse the mounted
Hugging Face cache:

```bash
cd /home/url/PATH-G1/perception
docker build -t g1-perception-sam3:thor-jp71 -f docker/Dockerfile.thor .
./tools/run_thor.sh python tools/export_sam3_onnx.py \
  --output /models/sam3_trt/runtime4/sam3_4out.onnx --device cpu
./tools/run_thor.sh bash tools/build_sam3_trt.sh \
  /models/sam3_trt/runtime4/sam3_4out.onnx \
  /models/sam3_trt/sam3_fp16_4out.engine
```

TensorRT 10 uses its `--fp16` builder mode. TensorRT 11 uses NVIDIA ModelOpt
AutoCast to make FP16 types explicit before building the strongly typed engine.
The export omits the unused `semantic_seg` branch and keeps the four outputs
required by the instance pipeline. CUDA Graph replay is enabled by default
after one warm-up enqueue; set `G1_SAM3_CUDA_GRAPH=0` to use direct TensorRT
enqueue. The web server retains `--backend pytorch` as an immediate fallback.

Run one prompt per inference for the lowest latency:

```bash
./tools/run_thor.sh python -m seg_web_node.web_server \
  --config /workspace/perception/config/pipeline.yaml \
  --backend tensorrt \
  --concepts object \
  --allow-unaligned \
  --host 0.0.0.0 \
  --port 8080
```

Each additional comma-separated concept is a separate full TensorRT inference,
so a seven-concept vocabulary is not expected to have single-prompt latency.
Use the web prompt control to choose the smallest useful concept set.

Validated on this board in the 120 W profile:

| Path | Same 640x360 image, one `floor` prompt |
| --- | ---: |
| Existing PyTorch backend | 522 ms mean |
| TensorRT Python backend | 303 ms mean, 297 ms p50 |
| Raw TensorRT engine | 294 ms mean, 293 ms p50 |

The full mock RGB-D/web path produced the expected mask and depth with zero
decode/pairing errors. To reproduce the reference project's MAXN condition,
first verify cooling and power-delivery headroom, then run on the host:

```bash
nvpmodel -q
sudo nvpmodel -m 0
sudo jetson_clocks
```

Mode `0` is MAXN in this Thor's `/etc/nvpmodel.conf`. A mode change can require
a reboot; follow the message printed by `nvpmodel`. This deployment does not
change the board power profile automatically.

Before that launch, replace all null camera intrinsics in
`config/camera_info.yaml` and provide `T_torso_cam` in
`config/extrinsics.yaml` if torso-frame grasp output is required.

## Validation recorded on this board

On 2026-07-21 the final image passed:

- ARM64/Python 3.12 and `pip check`;
- PyTorch 2.13 with a real CUDA kernel on Thor `sm_110`;
- SAM3 image/video builder and processor imports plus signature inspection;
- LCM multicast handle creation on the configured URL;
- FastAPI, Uvicorn, OpenCV, SciPy, YAML, and WebSocket imports;
- cached DINOv2 ViT-S/14 patch-token inference (`1x256x384`), averaging
  approximately 3.5 ms over 10 synthetic 224x224 trials.

The NGC image prints a CUDA 13.3-on-driver-13.0 minor-compatibility warning.
The actual `sm_110` kernel tests above pass, so this is not currently a runtime
failure. Re-run `tools/verify_environment.py` after any JetPack or image update.
