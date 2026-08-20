# Remote perception paths

`*_head_d435i_legacy` modules and the `run_thor_remote_*` launchers preserve
the former G1-LCM/head-D435i pipeline for ablation only.

Production wrist perception is explicit and separate:

```bash
# On Thor after rs-enumerate preflight identified the four UVC nodes and two USB paths.
# From the recorded Thor enumeration: Z16/UYVY are 0/2 at port 1-4.2.4 and
# 6/8 at port 1-4.2.3. Re-run topology/profile checks after any reconnect.
export D405_VIDEO_DEVICES=/dev/video0,/dev/video2,/dev/video6,/dev/video8
export D405_USB_DEVICES=/dev/bus/usb/001/038,/dev/bus/usb/001/039
./tools/run_thor_wrist_d405_gateway_v11.sh --preflight
./tools/run_thor_wrist_d405_gateway_v11.sh --lcm-url "$PATH_LCM_URL"

# On the remote SAM3 host.
./tools/run_remote_sam3_wrist_d405_v11.sh --wifi-lcm-url "$WIFI_LCM_URL" \
  --backend tensorrt --trt-engine /models/sam3_trt/remote.engine
```

The v11 gateway opens D405s directly by serial and publishes only
`PATH_V11_WRIST_*` camera messages. It never subscribes to head RGB, depth, or
calibration channels. It requires both configured cameras by default and will
not kill a conflicting V4L2 owner such as `teleimager-serv`.

Before first capture, run `python3 tools/query_d405_intrinsics.py --ssh
url@thor --json --raw`, confirm left/right physical mounting with a covered
lens, and validate the CAD-only wrist extrinsics with a hand-eye procedure.
