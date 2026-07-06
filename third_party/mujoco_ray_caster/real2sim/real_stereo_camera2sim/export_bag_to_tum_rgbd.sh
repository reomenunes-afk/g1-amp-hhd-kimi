#!/usr/bin/env bash
set -eo pipefail

WORKFLOW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROS_WS="${ROS_WS:-/home/albusgive2/deep_camear_ros}"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 path_to_rosbag output_dataset_dir [extra exporter args]" >&2
  exit 1
fi

source /opt/ros/humble/setup.bash
source "$ROS_WS/install/setup.bash"

exec python3 "$WORKFLOW_DIR/export_bag_to_tum_rgbd.py" "$@"
