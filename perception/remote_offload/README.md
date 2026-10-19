# Remote wrist SAM3

The production remote path pairs two wrist D405 cameras on Thor with SAM3
inference on a separate x86/GPU host. The Thor gateway keeps camera ownership,
timestamp-pairs each result with `G1_ROBOT_STATE`, runs belief assistance, and
publishes a selected `path_static_ee_target` for PATH's nonlinear arm IK.

Start the Thor gateway with the current deployment wrapper:

```bash
./tools/run_thor_wrist_d405_gateway_v11.sh --preflight
./tools/run_thor_wrist_d405_gateway_v11.sh \
  --g1-lcm-url "$G1_LCM_URL" --wifi-lcm-url "$WIFI_LCM_URL" \
  --belief-inference --belief-input-source operator_command \
  --ee-target-publishing \
  --ee-target-endpoint tcp://127.0.0.1:5564 \
  --ee-target-topic path_static_ee_target
```

On the x86 inference host, build the remote TensorRT worker image and place a
TensorRT engine built for that host at `models/sam3_trt/rtx4060/` (or set
`SAM3_TRT_ENGINE_HOST` to its host path). The image installs the PyTorch,
TensorRT, and Transformers runtime; the TensorRT engine is a separate model
artifact and is not stored in Git.

```bash
docker build -f docker/Dockerfile.remote-sam3-trt \
  -t path-g1-remote-sam3-trt:rtx4060 .
```

Run the worker with the same Wi-Fi LCM URL:

```bash
SAM3_TRT_ENGINE_HOST="$SAM3_ENGINE" \
  ./tools/run_remote_sam3_wrist_d405_v11.sh \
  --wifi-lcm-url "$WIFI_LCM_URL"
```

The worker handles masks and segmentation geometry. The Thor gateway retains
robot-state pairing, surface/contact belief, left/right candidate ownership,
and target publication. Both D405 extrinsics must be calibrated for the
installed mounts before target-driven control is used. The gateway publishes
targets on port 5564; the selected-hand heartbeat uses port 5567. LCM multicast
groups and all runtime ports are listed in [`../../docs/ports_and_topics.md`](../../docs/ports_and_topics.md).
