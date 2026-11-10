#!/usr/bin/env bash
set -u

BOOST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$BOOST_ROOT/configs/teleop/local.env"

interface="${G1_NETWORK_INTERFACE:-eth0}"
thor_ip="${BOOST_THOR_G1_LINK_IP:-192.168.123.2}"
g1_ip="${G1_TELEIMAGER_HOST:-192.168.123.164}"
local_ipv4="$(ip -o -4 addr show 2>/dev/null | awk '{split($4,a,"/"); print a[1]}')"
peer_ip="${1:-}"
if [[ -z "$peer_ip" ]]; then
  case " $local_ipv4 " in
    *" $thor_ip "*) peer_ip="$g1_ip" ;;
    *" $g1_ip "*) peer_ip="$thor_ip" ;;
    *) peer_ip="$g1_ip" ;;
  esac
fi

echo "=== BOOST teleop network diagnostics ==="
printf 'host=%s interface=%s peer=%s\n' "$(hostname)" "$interface" "$peer_ip"
printf 'PATH_LCM_URL=%s\nWIFI_LCM_URL=%s\n' "$PATH_LCM_URL" "$WIFI_LCM_URL"
printf 'Expected wired peers: Thor=%s G1=%s\n' "$thor_ip" "$g1_ip"
printf 'Unitree DDS: domain=0 interface=%s (DDS ports are participant-assigned)\n' "$interface"

echo
echo "--- IPv4 addresses, route, and peer reachability ---"
ip -brief -4 addr 2>/dev/null || true
ip route get "$peer_ip" 2>&1 || true
path_route="$(ip route get 239.255.76.67 2>&1 || true)"
printf '%s\n' "$path_route"
ip route get 239.255.77.67 2>&1 || true
path_route_dev="$(awk '{for (i=1; i<=NF; i++) if ($i == "dev") {print $(i+1); exit}}' <<<"$path_route")"
interface_src="$(ip -o -4 addr show dev "$interface" 2>/dev/null | awk 'NR == 1 {split($4, a, "/"); print a[1]}')"
if [[ -n "$path_route_dev" && "$path_route_dev" != "$interface" ]]; then
  echo "WARNING: PATH LCM multicast route uses $path_route_dev; wired teleop expects $interface."
  if [[ -n "$interface_src" ]]; then
    echo "On this host, apply for the current boot with: sudo ip route replace 239.255.76.67/32 dev $interface src $interface_src"
  else
    echo "On this host, apply for the current boot with: sudo ip route replace 239.255.76.67/32 dev $interface"
  fi
elif [[ -n "$path_route_dev" ]]; then
  echo "PATH LCM multicast route is on $path_route_dev."
fi
ip neigh show dev "$interface" 2>/dev/null | awk -v peer="$peer_ip" '$1 == peer {print}'
if command -v ping >/dev/null 2>&1; then
  ping -I "$interface" -c 2 -W 1 "$peer_ip" 2>&1 || true
fi

echo
echo "--- Configured TCP/UDP endpoints ---"
cat <<EOF
Galaxy XR HTTPS/WebSocket: Thor TCP 8012
Galaxy XR hand pose PUB -> PATH and hand-MPPI SUB: Thor TCP 5566 (vr_pose_debug)
PATH QP telemetry PUB: Thor TCP 5560
PATH state-command SUB: Thor TCP 5563
PATH static-target SUB: Thor TCP 5564
PATH shared-hand PUB: Thor TCP 5567
G1 Teleimager config request: G1 ZMQ TCP 60000
G1 Teleimager preview streams: ZMQ TCP 55555-55557
G1 Teleimager recording streams: ZMQ TCP 55558-55560
G1 Teleimager WebRTC signaling/streams: TCP 60001-60003 (if enabled)
DreamWaQ remote velocity input: G1 UDP ${g1_ip}:15001, allowed Thor sender ${thor_ip}
Thor <-> G1 PATH command/state: LCM multicast UDP 7667, group 239.255.76.67
Thor <-> perception/recorder: LCM multicast UDP 7667, group 239.255.77.67
TCP 5565: unassigned by the Galaxy XR hand teleop path
EOF
for config in \
  "$BOOST_ROOT/external/dreamwaq_g1_29dof/deploy/robot_control/configs/g1_homie.yaml" \
  "$BOOST_ROOT/configs/sim/dreamwaq_g1_mujoco.yaml"; do
  if [[ -f "$config" ]]; then
    echo "DreamWaQ UDP configuration from $config:"
    awk '/cmd_udp_(bind_host|port|allowed_host):/ {print "  " $0}' "$config"
  fi
done

echo
echo "--- Selected listeners on this host ---"
if command -v ss >/dev/null 2>&1; then
  echo "TCP:"
  ss -ltnp 2>/dev/null | awk 'NR == 1 || $4 ~ /:(8012|5560|5563|5564|5566|5567|55555|55556|55557|55558|55559|55560|60000|60001|60002|60003)$/ {print}'
  echo "UDP:"
  ss -lunp 2>/dev/null | awk 'NR == 1 || $4 ~ /:(7667|15001)$/ {print}'
  echo "Established TCP to peer (if any):"
  ss -tnp state established 2>/dev/null | awk -v peer="$peer_ip" 'NR == 1 || index($4,peer ":") || index($5,peer ":") {print}'
else
  echo "ss is unavailable; install iproute2 to inspect socket owners."
fi

echo
echo "--- Relevant runtime processes ---"
if command -v pgrep >/dev/null 2>&1; then
  runtime_processes="$(pgrep -af 'g1_lowlevel_bridge|run_dreamwaq|path_state_machine_bridge|galaxy_xr_manager_thread_server|record_wrist_experiment_v11|hand_mppi' || true)"
  if [[ -n "$runtime_processes" ]]; then
    printf '%s\n' "$runtime_processes"
  else
    echo "No matching BOOST teleop, G1 bridge, or DreamWaQ processes are running."
    case " $local_ipv4 " in
      *" $g1_ip "*) echo "WARNING: G1 runtime is inactive; expected g1_lowlevel_bridge and run_dreamwaq.py." ;;
      *" $thor_ip "*) echo "Thor teleop is inactive; TCP listeners and PATH command packets are not expected yet." ;;
    esac
  fi
else
  echo "pgrep is unavailable."
fi

echo
echo "--- Multicast membership on $interface ---"
ip maddr show dev "$interface" 2>&1 || true

echo
echo "--- Live LCM channel probe (5 seconds) ---"
"${PATH_PYTHON:-python}" - "$PATH_LCM_URL" "$WIFI_LCM_URL" <<'PY'
import sys
import time

try:
    import lcm
except Exception as error:
    print(f"LCM probe unavailable in this Python ({error}); run under the teleop Pixi/Conda environment.")
    raise SystemExit(0)

channels = {
    "PATH": ("G1_ROBOT_STATE", "G1_PATH_UPPER_COMMAND"),
    "WIFI": (
        "PATH_V11_WRIST_IPOPT_VISUAL",
        "PATH_V11_WRIST_BELIEF_TELEMETRY",
        "PATH_V11_WRIST_EXPERIMENT_RECORD_CONTROL",
    ),
}
urls = {"PATH": sys.argv[1], "WIFI": sys.argv[2]}
counts = {name: {channel: 0 for channel in selected} for name, selected in channels.items()}
buses = {}
for name in ("PATH", "WIFI"):
    bus = lcm.LCM(urls[name])
    buses[name] = bus
    for channel in channels[name]:
        bus.subscribe(channel, lambda _channel, _data, n=name, c=channel: counts[n].__setitem__(c, counts[n][c] + 1))

deadline = time.monotonic() + 5.0
while time.monotonic() < deadline:
    for bus in buses.values():
        bus.handle_timeout(40)

for name in ("PATH", "WIFI"):
    print(f"{name} {urls[name]}")
    for channel, count in counts[name].items():
        print(f"  {channel}: {count} packets in 5s")
if counts["PATH"]["G1_ROBOT_STATE"] == 0:
    print("NO G1_ROBOT_STATE: check the G1 adapter's LCM publish log, PATH_LCM_URL, eth0 multicast route, and firewall.")
if counts["PATH"]["G1_PATH_UPPER_COMMAND"] == 0:
    print("NO G1_PATH_UPPER_COMMAND: this may be normal until Thor PATH is running and publishing arm references.")
if counts["WIFI"]["PATH_V11_WRIST_IPOPT_VISUAL"] == 0:
    print("NO PATH_V11_WRIST_IPOPT_VISUAL: expected when the Thor PATH bridge is not running.")
if counts["WIFI"]["PATH_V11_WRIST_EXPERIMENT_RECORD_CONTROL"] == 0:
    print("NO recorder-control packet: normal until terminal s is pressed in an active teleop session.")
PY
