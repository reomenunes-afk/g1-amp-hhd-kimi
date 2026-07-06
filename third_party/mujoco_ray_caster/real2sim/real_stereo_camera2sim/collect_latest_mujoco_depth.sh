#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAYCASTER_ROOT="$(cd "$PROJECT_DIR/../.." && pwd)"
DATASETS_DIR="$PROJECT_DIR/datasets"

DATASET_NAME=""
if [[ $# -gt 0 && "${1:-}" != -* ]]; then
  DATASET_NAME="$1"
fi
if [[ -n "$DATASET_NAME" ]]; then
  if [[ "$DATASET_NAME" = /* ]]; then
    DATASET_DIR="$DATASET_NAME"
  else
    DATASET_DIR="$DATASETS_DIR/$DATASET_NAME"
  fi
  shift
else
  DATASET_DIR="$(find "$DATASETS_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)"
fi

if [[ -z "${DATASET_DIR:-}" || ! -d "$DATASET_DIR" ]]; then
  echo "No dataset directory found under: $DATASETS_DIR" >&2
  exit 1
fi

TRAJ="$DATASET_DIR/CameraTrajectory.txt"
if [[ ! -f "$TRAJ" ]]; then
  echo "Trajectory not found: $TRAJ" >&2
  echo "Run process_latest_recording.sh first." >&2
  exit 1
fi

for arg in "$@"; do
  case "$arg" in
    --viewer|--fixed-view|--viewer-fps)
      echo "The C++ collector does not support viewer mode yet." >&2
      echo "Use replay_latest_mujoco.sh without viewer flags for data collection." >&2
      exit 1
      ;;
    --render-depth|--no-render-depth|--render-depth-noise-std|--render-depth-noise-seed|--render-depth-clip-min|--render-depth-clip-max)
      echo "render-depth arguments are obsolete." >&2
      echo "The third depth stream now comes from tracked_aux_depth_raycaster in mujoco_camera_replay.xml." >&2
      exit 1
      ;;
  esac
done

OUT_DIR="$DATASET_DIR/mujoco_depth"

COLLECTOR_BIN="$PROJECT_DIR/bin/mujoco_rgbd_collector"
if [[ ! -x "$COLLECTOR_BIN" ]]; then
  "$PROJECT_DIR/build_mujoco_rgbd_collector.sh"
fi

exec "$COLLECTOR_BIN" \
  "$TRAJ" \
  --output "$OUT_DIR" \
  --xml "$PROJECT_DIR/mujoco_camera_replay.xml" \
  --plugin "$RAYCASTER_ROOT/lib/libsensor_raycaster.so" \
  "$@"
