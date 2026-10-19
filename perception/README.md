# PATH perception

Perception provides the arm controller with synchronized RGB-D observations,
SAM3 masks, fitted surface geometry, and belief-assisted end-effector targets.
It does not command robot joints.

## Runtime pieces

- `seg_web_node/`: RGB-D synchronization, segmentation, robot self-filtering,
  surface/point geometry, belief visualization, and target publication.
- `remote_offload/`: paired wrist D405 capture on Thor, remote SAM3 inference,
  timestamped results, and belief/target integration.
- `config/pipeline.yaml`: current Thor runtime configuration.
- `config/extrinsics.yaml` and `remote_offload/wrist_d405_protocol_v11.py`:
  camera frames and the wrist observation protocol.

The Thor web server reads G1 state from `G1_ROBOT_STATE` and publishes selected
end-effector targets on ZMQ TCP 5564 as `path_static_ee_target`. Its operator
view defaults to HTTP port 8080. See the central [port registry](../docs/ports_and_topics.md).

## Thor web process

Run inside the configured Thor perception environment:

```bash
python -m seg_web_node.web_server \
  --config config/pipeline.yaml \
  --host 0.0.0.0 --port 8080
```

## Remote wrist SAM3

The D405 gateway and SAM3 worker are separate processes. The Thor gateway owns
the cameras and publishes paired frames; the remote worker performs inference
and returns masks/geometry over the configured LCM groups. Belief assistance
and target selection remain on Thor, where G1 state and fitted surface data are
available together. Current launch commands and frame ownership are in
[`remote_offload/README.md`](remote_offload/README.md).

The worker is an inference service only: it has no Unitree DDS publisher and
cannot write `rt/arm_sdk` or `rt/lowcmd`.

For a remote browser view of the wrist stream, start
`tools/run_remote_web_wrist_d405_v11.sh`; it serves HTTP on port 8082 by default.
