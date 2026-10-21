#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
perception_dir="$(cd -- "${script_dir}/.." && pwd)"
image="${G1_PERCEPTION_D405_IMAGE:-g1-perception-sam3:d435i-thor-jp71}"
container_name="${G1_PERCEPTION_CONTAINER_NAME:-g1-perception-wrist-d405-v11}"

auto_devices=0
remaining_args=()
for arg in "$@"; do
    case "${arg}" in
        --auto-devices)
            auto_devices=1
            ;;
        *)
            remaining_args+=("${arg}")
            ;;
    esac
done
set -- "${remaining_args[@]}"

detect_d405_devices() {
    local video_sys current parent vendor product bus device local_video local_usb
    local -a videos=() usb_nodes=()
    declare -A seen_video=() seen_usb=()

    # Do not match the truncated V4L2 display name.  Instead walk every video
    # node to its physical USB parent and accept only Intel D405 (8086:0b5b).
    for video_sys in /sys/class/video4linux/video*; do
        [[ -r "${video_sys}/name" ]] || continue
        current="$(readlink -f "${video_sys}/device" 2>/dev/null || true)"
        while [[ -n "${current}" && "${current}" != "/" && "${current}" != "/sys" ]]; do
            vendor="$(cat "${current}/idVendor" 2>/dev/null || true)"
            product="$(cat "${current}/idProduct" 2>/dev/null || true)"
            if [[ "${vendor}" == "8086" && "${product}" == "0b5b" ]]; then
                local_video="/dev/$(basename "${video_sys}")"
                [[ -c "${local_video}" ]] || break
                if [[ -z "${seen_video[${local_video}]:-}" ]]; then
                    videos+=("${local_video}")
                    seen_video[${local_video}]=1
                fi
                bus="$(cat "${current}/busnum" 2>/dev/null || true)"
                device="$(cat "${current}/devnum" 2>/dev/null || true)"
                if [[ "${bus}" =~ ^[0-9]+$ && "${device}" =~ ^[0-9]+$ ]]; then
                    local_usb="/dev/bus/usb/$(printf '%03d' "${bus}")/$(printf '%03d' "${device}")"
                    if [[ -e "${local_usb}" && -z "${seen_usb[${local_usb}]:-}" ]]; then
                        usb_nodes+=("${local_usb}")
                        seen_usb[${local_usb}]=1
                    fi
                fi
                break
            fi
            parent="$(dirname "${current}")"
            [[ "${parent}" != "${current}" ]] || break
            current="${parent}"
        done
    done

    if (( ${#videos[@]} == 0 || ${#usb_nodes[@]} != 2 )); then
        echo "D405 auto-detect expected V4L2 nodes and exactly two USB devices; found videos=${videos[*]:-none} usb=${usb_nodes[*]:-none}" >&2
        exit 2
    fi
    D405_VIDEO_DEVICES="$(IFS=,; echo "${videos[*]}")"
    D405_USB_DEVICES="$(IFS=,; echo "${usb_nodes[*]}")"
    export D405_VIDEO_DEVICES D405_USB_DEVICES
    echo "Auto-detected D405_VIDEO_DEVICES=${D405_VIDEO_DEVICES}" >&2
    echo "Auto-detected D405_USB_DEVICES=${D405_USB_DEVICES}" >&2
}

# Export comma-separated nodes found during Thor preflight, for example:
# D405_VIDEO_DEVICES=/dev/video0,/dev/video1 D405_USB_DEVICES=/dev/bus/usb/001/038,/dev/bus/usb/001/039
# The launcher intentionally refuses to grant --privileged or a blanket /dev
# mount.  teleimager-serv must be stopped by the operator before launch.
if (( auto_devices )); then
    detect_d405_devices
fi
if [[ -z "${D405_VIDEO_DEVICES:-}" || -z "${D405_USB_DEVICES:-}" ]]; then
    echo "Set D405_VIDEO_DEVICES and D405_USB_DEVICES, or pass --auto-devices; refusing broad device access." >&2
    exit 2
fi
IFS=',' read -r -a video_devices <<< "${D405_VIDEO_DEVICES}"
IFS=',' read -r -a usb_devices <<< "${D405_USB_DEVICES}"
device_args=()
for device in "${video_devices[@]}" "${usb_devices[@]}"; do
    [[ -e "${device}" ]] || { echo "Missing D405 device: ${device}" >&2; exit 2; }
    device_args+=(--device "${device}:${device}:rwm")
done
if command -v fuser >/dev/null 2>&1; then
    # Check on the host: a normal Docker PID namespace cannot report a host
    # service (for example teleimager-serv) that already opened D405 UVC nodes.
    # Never terminate that service automatically.
    if fuser -v "${video_devices[@]}"; then
        echo "A process already owns a configured D405 V4L2 node. Stop it manually, then retry." >&2
        exit 2
    fi
fi
# Capture and JPEG transport do not use CUDA. Omitting GPU passthrough also
# avoids unrelated CUDA compatibility failures on a Thor driver mismatch.
# The host udev ACL for USB bus nodes does not survive Docker's user namespace
# consistently. Run this dedicated capture process as root, but expose only
# the explicitly supplied D405 nodes (never --privileged or all of /dev).
exec docker run --rm --network host --ipc host \
    --name "${container_name}" \
    --user 0:0 \
    -e D405_VIDEO_DEVICES \
    -e D405_USB_DEVICES \
    -v "${perception_dir}:/workspace/perception" \
    -v /run/udev:/run/udev:ro \
    -v /sys:/sys:ro \
    "${device_args[@]}" \
    -w /workspace/perception "${image}" \
    python -m remote_offload.thor_wrist_d405_gateway_v11 "$@"
