#!/usr/bin/env bash
set -euo pipefail
SLAM_ARGS=("$@")
set --
set +u
source /opt/ros/noetic/setup.bash
source /home/unitree/nature_task/nature_ws/devel/setup.bash
set -u
export ROS_MASTER_URI="${ROS_MASTER_URI:-http://192.168.0.19:11311}"
export ROS_IP="${ROS_IP:-192.168.0.19}"
unset ROS_HOSTNAME
# OMP_THREAD_LIMIT also caps code that explicitly requests eight threads.
export OMP_NUM_THREADS=2 OMP_THREAD_LIMIT=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec taskset -c 2-3 roslaunch url_lidar_slam 3_maploc.launch "${SLAM_ARGS[@]}"
