# Remote wrist SAM3

The production path pairs two wrist D405 cameras on Thor with SAM3 inference
on a separate x86/GPU host. Thor owns camera capture, timestamp-pairs results
with `G1_ROBOT_STATE`, runs surface/belief processing, and publishes selected
`path_static_ee_target` messages to PATH's nonlinear arm IK.

## Thor camera preflight

Thor needs a PATH-G1 checkout and the D405 capture image
`g1-perception-sam3:d435i-thor-jp71` (override with
`G1_PERCEPTION_D405_IMAGE`). This is the RealSense capture image; the remote
SAM3 inference image is a separate image. Stop any existing process that owns
the D405 cameras before running the preflight.

The launcher requires explicit device nodes. `--auto-devices` detects and
passes only the two configured D405 cameras, so include it on each invocation:

```bash
cd PATH-G1/perception
docker image inspect g1-perception-sam3:d435i-thor-jp71
./tools/run_thor_wrist_d405_gateway_v11.sh --auto-devices --preflight
./tools/run_thor_wrist_d405_gateway_v11.sh --auto-devices --capture-smoke
```

Confirm the configured serials map to the physical left and right wrists. The
current mapping is provisional until verified on the installed robot.

## Remote SAM3 worker

The inference host needs a PATH-G1 checkout, the image
`path-g1-remote-sam3-trt:rtx4060` (override with `SAM3_DOCKER_IMAGE`), and a
TensorRT engine built for that host's GPU and runtime. The engine is a separate
file; installing the Docker image alone is not sufficient. The default host
path is `perception/models/sam3_trt/rtx4060/sam3_fp16_4out.engine`.

```bash
cd PATH-G1/perception
docker image inspect path-g1-remote-sam3-trt:rtx4060
test -f models/sam3_trt/rtx4060/sam3_fp16_4out.engine
./tools/run_remote_sam3_wrist_d405_v11.sh \
  --wifi-lcm-url 'udpm://239.255.77.67:7667?ttl=1'
```

Set `SAM3_TRT_ENGINE_HOST` if the engine is elsewhere. The worker also mounts
the PATH-G1 checkout at runtime, so the checkout is needed on the inference
host even when the image is already installed.

## Thor gateway and target publishing

Start the worker on the inference host. On Thor, run the gateway with target
publishing disabled for the initial camera/SAM3/belief integration check:

```bash
cd PATH-G1/perception
./tools/run_thor_wrist_d405_gateway_v11.sh --auto-devices \
  --g1-lcm-url 'udpm://239.255.76.67:7667?ttl=1' \
  --wifi-lcm-url 'udpm://239.255.77.67:7667?ttl=1' \
  --belief-inference --belief-input-source operator_command \
  --no-ee-target-publishing
```

The current wrist transforms are CAD-derived estimates, not measured
hand-eye calibrations. Calibrate both transforms and verify the camera-to-hand
assignment before enabling target-driven robot motion. After that verification,
restart the gateway with `--ee-target-publishing` and the PATH state machine
listening on TCP 5564. Its selected-hand heartbeat uses TCP 5567.

Thor and the inference host must exchange LCM multicast on
`239.255.77.67:7667`; G1 and Thor must exchange `G1_ROBOT_STATE` on
`239.255.76.67:7667`. The configured multicast TTL is 1, so these routes
must work within the network's multicast domain. See the
[port registry](../../docs/ports_and_topics.md).
