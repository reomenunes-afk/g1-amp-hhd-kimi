#!/usr/bin/env bash
set -eo pipefail

ROS_WS="${ROS_WS:-/home/albusgive2/deep_camear_ros}"

source /opt/ros/humble/setup.bash
source "$ROS_WS/install/setup.bash"

echo "Camera topics:"
ros2 topic list | grep '^/camera/' || true

echo
echo "Expected topics for this workflow:"
echo "  /camera/color/image_raw"
echo "  /camera/depth/image_raw"
echo "  /camera/color/camera_info"
echo "  /camera/depth/camera_info"
