#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
perception_dir="$(cd -- "${script_dir}/.." && pwd)"
image="${G1_PERCEPTION_D405_IMAGE:-g1-perception-sam3:d435i-thor-jp71}"

# Export comma-separated nodes found during Thor preflight, for example:
# D405_VIDEO_DEVICES=/dev/video0,/dev/video1 D405_USB_DEVICES=/dev/bus/usb/001/038,/dev/bus/usb/001/039
# The launcher intentionally refuses to grant --privileged or a blanket /dev
# mount.  teleimager-serv must be stopped by the operator before launch.
if [[ -z "${D405_VIDEO_DEVICES:-}" || -z "${D405_USB_DEVICES:-}" ]]; then
    echo "Set D405_VIDEO_DEVICES and D405_USB_DEVICES from the Thor preflight; refusing broad device access." >&2
    exit 2
fi
IFS=',' read -r -a video_devices <<< "${D405_VIDEO_DEVICES}"
IFS=',' read -r -a usb_devices <<< "${D405_USB_DEVICES}"
device_args=()
for device in "${video_devices[@]}" "${usb_devices[@]}"; do
    [[ -e "${device}" ]] || { echo "Missing D405 device: ${device}" >&2; exit 2; }
    device_args+=(--device "${device}:${device}:rwm")
done
group_args=()
# ``--user uid:gid`` otherwise discards the caller's supplementary `video`,
# `plugdev`, or equivalent udev-access groups inside Docker.
for group_id in $(id -G); do
    group_args+=(--group-add "${group_id}")
done

# Capture and JPEG transport do not use CUDA. Omitting GPU passthrough also
# avoids unrelated CUDA compatibility failures on a Thor driver mismatch.
exec docker run --rm --network host --ipc host \
    --user "$(id -u):$(id -g)" \
    "${group_args[@]}" \
    -e D405_VIDEO_DEVICES \
    -e D405_USB_DEVICES \
    -v "${perception_dir}:/workspace/perception" \
    -v /run/udev:/run/udev:ro \
    -v /sys:/sys:ro \
    "${device_args[@]}" \
    -w /workspace/perception "${image}" \
    python -m remote_offload.thor_wrist_d405_gateway_v11 "$@"
