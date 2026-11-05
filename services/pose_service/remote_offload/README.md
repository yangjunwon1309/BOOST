# Remote wrist SAM3

The production path pairs two wrist D405 cameras on Thor with SAM3 inference
on a separate x86/GPU host. Thor owns camera capture, timestamp-pairs results
with `G1_ROBOT_STATE`, runs surface/belief processing, and publishes selected
`path_static_ee_target` messages to PATH's nonlinear arm IK.

## G1 Teleimager TIM2 wrist RGB path (image-only)

The selected G1 preset is
`external/teleimager/config/hbvcam_4m2214hd_g1.yaml`. It publishes TIM2 on
head port 55558 and wrist ports 55559 (left) and 55560 (right); wrist images
are 640×480. The relay reads these values from the selected Teleimager preset,
validates the TIM2 header, and publishes a version 2 wrist RGB message with the
original JPEG, camera sequence, G1 capture wall/monotonic time, and Thor
receive timestamp:

```bash
cd BOOST/services/pose_service
G1_TELEIMAGER_HOST=192.168.123.164 ./tools/run_thor_g1_teleimager_wrist_relay_v11.sh
```

Start the existing TensorRT container in RGB-only mode:

```bash
SAM3_INPUT_MODE=teleimager_rgb \
SAM3_DOCKER_IMAGE=g1-perception-sam3:thor-jp71 \
SAM3_TRT_ENGINE_HOST=/home/url/BOOST/services/pose_service/models/sam3_trt/sam3_fp16_4out.engine \
WIFI_LCM_URL='udpm://239.255.77.67:7667?ttl=1' \
./tools/run_remote_sam3_wrist_d405_v11.sh
```

The wrist image receiver accepts both legacy version 1 messages and version 2.
It uses the Thor receive timestamp for freshness checks and does not compare
monotonic clocks across G1 and Thor. Sequence gaps are logged by the relay.
This RGB-only path logs SAM3 detections; it does not publish masks to the viewer
or create targets. It has no depth channel and cannot produce calibrated 3-D
targets or arm IK commands. The D405 RGB-D mode remains the production geometry
path and is not replaced by Teleimager JPEG.

## Thor camera preflight

Thor needs a BOOST checkout and the D405 capture image
`g1-perception-sam3:d435i-thor-jp71` (override with
`G1_PERCEPTION_D405_IMAGE`). This is the RealSense capture image; the remote
SAM3 inference image is a separate image. Stop any existing process that owns
the D405 cameras before running the preflight.

The launcher requires explicit device nodes. `--auto-devices` detects and
passes only the two configured D405 cameras, so include it on each invocation:

```bash
cd BOOST/services/pose_service
docker image inspect g1-perception-sam3:d435i-thor-jp71
./tools/run_thor_wrist_d405_gateway_v11.sh --auto-devices --preflight
./tools/run_thor_wrist_d405_gateway_v11.sh --auto-devices --capture-smoke
```

Confirm the configured serials map to the physical left and right wrists. The
current mapping is provisional until verified on the installed robot.

## Remote SAM3 worker

The inference host needs a BOOST checkout, the image
`path-g1-remote-sam3-trt:rtx4060` (override with `SAM3_DOCKER_IMAGE`), and a
TensorRT engine built for that host's GPU and runtime. The engine is a separate
file; installing the Docker image alone is not sufficient. The default host
path is `services/pose_service/models/sam3_trt/rtx4060/sam3_fp16_4out.engine`.

```bash
cd BOOST/services/pose_service
docker image inspect path-g1-remote-sam3-trt:rtx4060
test -f models/sam3_trt/rtx4060/sam3_fp16_4out.engine
./tools/run_remote_sam3_wrist_d405_v11.sh \
  --wifi-lcm-url 'udpm://239.255.77.67:7667?ttl=1'
```

Set `SAM3_TRT_ENGINE_HOST` if the engine is elsewhere. The worker also mounts
the BOOST checkout at runtime, so the checkout is needed on the inference
host even when the image is already installed.

## Thor gateway and target publishing

Start the worker on the inference host. On Thor, run the gateway with target
publishing disabled for the initial camera/SAM3/belief integration check:

```bash
cd BOOST/services/pose_service
./tools/run_thor_wrist_d405_gateway_v11.sh --auto-devices \
  --g1-lcm-url 'udpm://239.255.76.67:7667?ttl=1' \
  --wifi-lcm-url 'udpm://239.255.77.67:7667?ttl=1' \
  --belief-inference --belief-input-source operator_command \
  --no-ee-target-publishing
```

The current CAD-derived wrist transforms are the accepted values for this
installation; a separate hand-eye calibration is not a prerequisite. Verify
that the configured camera serials correspond to the physical left and right
wrists and inspect the resulting targets. Then restart the gateway with
`--ee-target-publishing` and the PATH state machine listening on TCP 5564. Its
selected-hand heartbeat uses TCP 5567. Recheck the transforms if a camera mount
or holder changes.

Thor and the inference host must exchange LCM multicast on
`239.255.77.67:7667`; G1 and Thor must exchange `G1_ROBOT_STATE` on
`239.255.76.67:7667`. The configured multicast TTL is 1, so these routes
must work within the network's multicast domain. See the
[port registry](../../../docs/ports_and_topics.md).
