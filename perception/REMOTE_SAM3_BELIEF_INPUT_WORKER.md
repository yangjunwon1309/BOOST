# RTX 4060 Remote SAM3 Belief-Input Worker (head-D435i legacy ablation)

This worker remains for head-D435i ablation. The wrist-only production worker
is `remote_offload.remote_sam3_wrist_d405_worker_v11`.

This runtime keeps SAM3 and all Python dependencies inside a laptop-specific
Docker image. It publishes perception geometry only; it contains no robot
command, shared-hand, EE-target, DDS, Thor TensorRT, or JetPack publisher.

Build from the `perception` directory:

The image uses CUDA 12.6.3 with PyTorch 2.7.1 (`cu126`). The host must provide
only a compatible NVIDIA driver and NVIDIA Container Toolkit; do not install
the CUDA Toolkit, cuDNN, PyTorch, or SAM3 into the host/root environment.
CUDA 12.x minor-version compatibility starts at Linux driver 525.60.13, while
the native CUDA 12.6.3 driver floor is 560.35.05. This RTX 4060 laptop is first
validated with its previously working 535 driver branch. `nvidia-smi` and a
small CUDA kernel test must succeed before installing/configuring the NVIDIA
Container Toolkit or running the worker.

```bash
docker build -t path-g1-remote-sam3:rtx4060 \
  -f docker/Dockerfile.remote-sam3 .
```

Verify CUDA/SAM3 imports without loading the checkpoint:

```bash
docker run --rm --gpus all \
  path-g1-remote-sam3:rtx4060 \
  python tools/verify_remote_sam3_environment.py
```

Run the worker. Keep Hugging Face credentials in the shell/environment and
cache; never add them to the repository:

```bash
docker run --rm --gpus all --network host \
  -e HF_TOKEN \
  -v "$PWD/models:/models:ro" \
  -v "$PWD/config:/workspace/perception/config:ro" \
  -v "$HOME/.cache/huggingface:/cache/huggingface" \
  -v "$HOME/.cache/torch:/cache/torch" \
  path-g1-remote-sam3:rtx4060 \
  python -m remote_offload.remote_sam3_belief_input_worker_head_d435i_legacy \
    --wifi-lcm-url 'udpm://239.255.77.67:7667?ttl=1' \
    --config config/pipeline.yaml \
    --backend pytorch \
    --concepts bottle \
    --inference-hz 8
```

Run the viewer and perception prompt controller from the same isolated image:

```bash
docker run --rm --network host \
  -v "$PWD:/workspace/perception:ro" \
  path-g1-remote-sam3:rtx4060 \
  python -m remote_offload.remote_web_server_head_d435i_legacy \
    --wifi-lcm-url 'udpm://239.255.77.67:7667?ttl=1' \
    --host 0.0.0.0 --port 8081
```

Open `http://127.0.0.1:8081`, enter one or more comma-separated labels, and
press **Apply** (or Enter). The web process sends a versioned perception-only
command on `PATH_REMOTE_CONCEPT_CONTROL`. The worker applies it between frames,
clears stale semantic tracks, and reports the active labels in its heartbeat.
The initial `--concepts` value remains the startup fallback. Each additional
SAM3 text label adds another prompt pass, so use one label when minimum latency
is important.

Run the authoritative gateway on Thor with the existing Thor container/runtime:

```bash
./tools/run_thor_remote_gateway.sh \
  --g1-lcm-url "$G1_LCM_URL" \
  --wifi-lcm-url "$WIFI_LCM_URL" \
  --config /workspace/perception/config/pipeline.yaml \
  --belief-inference
```

The worker subscribes only to `PATH_REMOTE_*` inputs on `WIFI_LCM_URL`. The
Thor gateway validates its result, owns `ParallelTascBelief`, mirrors bounded
diagnostic telemetry on `PATH_REMOTE_BELIEF_TELEMETRY`, and remains the only
process allowed to publish local control-side targets.
