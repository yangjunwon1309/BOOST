# G1 SAM3 perception environment (Jetson AGX Thor)

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

Once `seg_web_node.main` is implemented, launch it with:

```bash
./tools/run_thor.sh python -m seg_web_node.main \
  --config config/pipeline.yaml \
  --registry-dir registry_data \
  --port 8080
```

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
