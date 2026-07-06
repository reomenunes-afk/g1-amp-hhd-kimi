#!/usr/bin/env bash
set -eo pipefail

WORKFLOW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROS_WS="${ROS_WS:-/home/albusgive2/deep_camear_ros}"
OUT_ROOT="${1:-$WORKFLOW_DIR/recordings}"
STAMP="$(date +%Y%m%d_%H%M%S)"
BAG_PATH="$OUT_ROOT/gemini336l_$STAMP"

RGB_TOPIC="${RGB_TOPIC:-/camera/color/image_raw}"
DEPTH_TOPIC="${DEPTH_TOPIC:-/camera/depth/image_raw}"
RGB_INFO_TOPIC="${RGB_INFO_TOPIC:-/camera/color/camera_info}"
DEPTH_INFO_TOPIC="${DEPTH_INFO_TOPIC:-/camera/depth/camera_info}"

mkdir -p "$OUT_ROOT"

source /opt/ros/humble/setup.bash
source "$ROS_WS/install/setup.bash"

echo "Recording RGB-D bag to: $BAG_PATH"
echo "Press Ctrl-C to stop recording."

exec ros2 bag record -o "$BAG_PATH" \
  "$RGB_TOPIC" \
  "$DEPTH_TOPIC" \
  "$RGB_INFO_TOPIC" \
  "$DEPTH_INFO_TOPIC"
