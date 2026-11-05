#!/usr/bin/env bash
# run_dreamwaq_autonomy.sh (dwcontrol) plus a switch: Y pressed and released
# on its own hands the standing robot to NEW2-3 backpack for the manipulation
# tasks and back, without passing through damping.  Everything else is the
# autonomy launcher's own: NEW3-4 backpack, ROS bridge input on
# 127.0.0.1:15002, CPU partition watchdog and layout, L1+Y.
#   START  default-pose ramp        A  start DreamWaQ (NEW3-4, standing)
#   X      standing <-> walking     Y  NEW3-4 <-> NEW2-3 (stand still first)
#   L1+Y   Unitree regular walking  Select  damping and exit
# Y to NEW2-3 holds the bridge at zero: NEW2-3 has no standing mode.  Y back
# needs the legs measured still and rt/arm_sdk quiet (the mission client gone);
# the arms then return to the default pose, NEW3-4 starts in standing, and the
# bridge stays held, so only the sticks drive until Thor releases it.
#
# Thor's ./nature demo asks for the same things over UDP 15003 (remote_mode.py):
# walking/standing for X, switch for Y there, hold_external, which zeroes the
# bridge (the sticks still drive) and has to come before the switch, and
# release_external, which the next run's preflight sends in standing.  It also
# receives the mode, the export and the bridge state/reason, which is how Thor
# sees the arrival.  Only Thor (192.168.123.2) is heard; every remote button
# keeps working.
# Extra arguments are appended, so a later flag overrides the one set here.
set -euo pipefail
REPO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
SWITCH_ARGS=(
  --switch-export "$REPO_DIR/checkpoints/new2_3_backpack_lower8cm_299999/exported"
  --switch-button Y
  --switch-max-vx 0.3 --switch-max-vy 0.20 --switch-max-yaw 0.4
  --switch-arm-smoothing-tau 0.04
  --switch-disable-lower-joint-target-clip
)
REMOTE_ARGS=(--remote-bind 192.168.123.164 --remote-allowed-host 192.168.123.2)
for arg in "$@"; do
  # A read-only shadow probe has no LowCmd to hand over.
  if [[ "$arg" == --shadow ]]; then
    SWITCH_ARGS=()
    REMOTE_ARGS=()
  fi
done
exec bash "$REPO_DIR/deploy/run_dreamwaq_autonomy.sh" "${SWITCH_ARGS[@]}" "${REMOTE_ARGS[@]}" "$@"
