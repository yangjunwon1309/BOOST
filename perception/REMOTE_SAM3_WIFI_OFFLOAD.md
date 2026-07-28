# Thor remote SAM3 Wi-Fi transport

Thor keeps G1 Ethernet LCM and all control paths local. The two new processes
only relay perception inputs/results; neither imports the state machine nor
publishes an EE target.

## Channels

| Direction | Channel | Payload |
|---|---|---|
| Thor → desktop | `PATH_REMOTE_RGB_FRAME` | Existing `CameraFrame` bytes |
| Thor → desktop | `PATH_REMOTE_DEPTH_FRAME` | Existing depth bytes |
| Thor → desktop | `PATH_REMOTE_CAMERA_CALIBRATION` | Existing calibration bytes |
| Thor → desktop | `PATH_REMOTE_ROBOT_STATE` | Existing robot-state bytes |
| Desktop → Thor | `PATH_REMOTE_PERCEPTION_RESULT` | Versioned MessagePack result |
| Thor local | `PATH_REMOTE_PERCEPTION_RESULT_LOCAL` | Validated unchanged result |

Use a dedicated Wi-Fi LCM URL/group, never the G1 Ethernet group. The Wi-Fi AP
must allow peer multicast. If it does not, replace only the Wi-Fi transport
with a unicast relay; preserve the channel names and MessagePack protocol.

## Thor launch

Run these in separate terminals. They do not start SAM3 or SONIC.

```bash
cd /home/url/PATH-G1/perception

bash tools/run_thor_remote_gateway.sh \
  --g1-lcm-url 'udpm://239.255.76.67:7667?ttl=1' \
  --wifi-lcm-url 'udpm://239.255.77.67:7667?ttl=1' \
  --forward-hz 5 --robot-state-hz 50

bash tools/run_thor_remote_result_receiver.sh \
  --g1-lcm-url 'udpm://239.255.76.67:7667?ttl=1' \
  --wifi-lcm-url 'udpm://239.255.77.67:7667?ttl=1' \
  --max-source-lag-ms 1500
```

Pin the multicast route for `239.255.76.67` to the G1 Ethernet interface and
the dedicated Wi-Fi group to `wlan0` before launch. Do not bridge interfaces.

## Desktop contract

The desktop worker subscribes to the four `PATH_REMOTE_*` input channels,
synchronizes RGB/depth by their original timestamps, and publishes a v1
MessagePack result defined in `remote_offload/protocol.py`. It hosts its own
webview; overlay images are not a control transport. The worker must publish
only the latest inference result and must never publish robot commands.
