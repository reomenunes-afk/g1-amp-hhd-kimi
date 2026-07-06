#!/usr/bin/env bash
set -eo pipefail

WORKFLOW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROS_WS="${ROS_WS:-/home/albusgive2/deep_camear_ros}"
PARAMS="${ORBBEC_PARAMS:-$WORKFLOW_DIR/gemini336l_orbbec_params.yaml}"

source /opt/ros/humble/setup.bash
source "$ROS_WS/install/setup.bash"

echo "Launching Orbbec Gemini 336L with params: $PARAMS"
echo "Extra ros2 launch args are passed through from this script."

exec ros2 launch orbbec_camera gemini_330_series.launch.py \
  config_file_path:="$PARAMS" \
  "$@"
