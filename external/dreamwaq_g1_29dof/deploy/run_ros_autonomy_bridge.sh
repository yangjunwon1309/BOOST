#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BRIDGE_ARGS=("$@")
# Catkin's sourced setup script reads the shell's positional arguments.
# Keep bridge options out of its environment-generation command.
set --
# Avoid the interactive ROS selector and Conda initialization in ~/.bashrc.
set +u
source /opt/ros/noetic/setup.bash
set -u
export ROS_MASTER_URI="${ROS_MASTER_URI:-http://192.168.0.19:11311}"
export ROS_IP="${ROS_IP:-192.168.0.19}"
unset ROS_HOSTNAME
exec taskset -c 0-3 /usr/bin/python3 -u "$SCRIPT_DIR/ros_autonomy_bridge.py" --print-commands "${BRIDGE_ARGS[@]}"
