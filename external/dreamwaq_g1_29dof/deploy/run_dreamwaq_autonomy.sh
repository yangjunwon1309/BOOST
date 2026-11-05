#!/usr/bin/env bash
set -euo pipefail
REPO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
CONTROLLER_ARGS=(--controller-switch)
for arg in "$@"; do
  if [[ "$arg" == --shadow ]]; then
    CONTROLLER_ARGS=()
    /usr/bin/python3 "$REPO_DIR/deploy/partition_autonomy_cpus.py" --check-idle
  fi
done
if [[ " $* " != *" --help "* && " $* " != *" -h "* ]]; then
  /usr/bin/python3 "$REPO_DIR/deploy/partition_autonomy_cpus.py" --apply
  CPU_LOG_DIR="$HOME/.cache/dreamwaq-autonomy-cpus"
  mkdir -p "$CPU_LOG_DIR"
  nohup /usr/bin/python3 "$REPO_DIR/deploy/partition_autonomy_cpus.py" --watch \
    >> "$CPU_LOG_DIR/watch.log" 2>&1 < /dev/null &
fi
# VS Code terminals inherit the editor's background mask; expand before the
# runtime assigns its own policy, control, and DDS thread affinities.
exec taskset -c 0-7 /home/unitree/miniconda3/envs/emer_tw/bin/dreamwaq-g1-29dof \
  --config "$REPO_DIR/deploy/robot_control/configs/g1_ros_autonomy.yaml" \
  --net eth0 \
  --export "$REPO_DIR/checkpoints/new3_4_backpack_440000/exported" \
  --locomotion-mode standing \
  --mode-toggle-button X \
  --max-vx 0.95 --max-vy 0.75 --max-yaw 0.75 \
  --arm-smoothing-tau 0.00 \
  --device cpu \
  --python-switch-interval-ms 1 --disable-cyclic-gc \
  --camera-cpus 0-1 --io-cpus 3-5 \
  --lowstate-reader-cpu 4 --arm-reader-cpu 3 --writer-cpu 5 \
  --policy-cpu 6 --control-cpu 7 \
  --policy-timeout-ms 15 --max-policy-timeouts 3 \
  "${CONTROLLER_ARGS[@]}" "$@"
