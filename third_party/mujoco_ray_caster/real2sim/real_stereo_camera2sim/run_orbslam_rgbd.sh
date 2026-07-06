#!/usr/bin/env bash
set -eo pipefail

WORKFLOW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORB_ROOT="${ORB_SLAM3_ROOT:-$WORKFLOW_DIR/third_party/ORB_SLAM3}"
COMPAT_LIB_DIR="$WORKFLOW_DIR/.runtime_libs"
PANGOLIN_LIB_DIR="${PANGOLIN_LIB_DIR:-/home/albusgive2/.local/pangolin-v0.6/lib}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 path_to_exported_tum_dataset [settings_yaml] [associations_txt]" >&2
  exit 1
fi

DATASET="$(realpath "$1")"
SETTINGS="$(realpath "${2:-$DATASET/orbslam3_gemini336l.yaml}")"
ASSOCIATIONS="$(realpath "${3:-$DATASET/associations.txt}")"
VOCAB="$ORB_ROOT/Vocabulary/ORBvoc.txt"
RGBD_EXE="$ORB_ROOT/Examples/RGB-D/rgbd_tum"

mkdir -p "$COMPAT_LIB_DIR"

if [[ ! -e "$COMPAT_LIB_DIR/librealsense2.so.2.56" ]]; then
  for candidate in \
    /opt/ros/humble/lib/x86_64-linux-gnu/librealsense2.so.2.57.7 \
    /opt/ros/humble/lib/x86_64-linux-gnu/librealsense2.so.2.57 \
    /opt/ros/humble/lib/x86_64-linux-gnu/librealsense2.so
  do
    if [[ -e "$candidate" ]]; then
      ln -sf "$candidate" "$COMPAT_LIB_DIR/librealsense2.so.2.56"
      break
    fi
  done
fi

if [[ -d "$PANGOLIN_LIB_DIR" ]]; then
  export LD_LIBRARY_PATH="$COMPAT_LIB_DIR:$PANGOLIN_LIB_DIR:${LD_LIBRARY_PATH:-}"
else
  export LD_LIBRARY_PATH="$COMPAT_LIB_DIR:${LD_LIBRARY_PATH:-}"
fi

if [[ ! -x "$RGBD_EXE" ]]; then
  echo "Missing executable: $RGBD_EXE" >&2
  exit 1
fi

if [[ ! -f "$VOCAB" || ! -f "$SETTINGS" || ! -f "$ASSOCIATIONS" ]]; then
  echo "Missing one of: vocabulary, settings, associations." >&2
  echo "VOCAB=$VOCAB" >&2
  echo "SETTINGS=$SETTINGS" >&2
  echo "ASSOCIATIONS=$ASSOCIATIONS" >&2
  exit 1
fi

echo "Running ORB-SLAM3 RGB-D on: $DATASET"
echo "Trajectories will be written into the dataset directory."

cd "$DATASET"
exec "$RGBD_EXE" "$VOCAB" "$SETTINGS" "$DATASET" "$ASSOCIATIONS"
