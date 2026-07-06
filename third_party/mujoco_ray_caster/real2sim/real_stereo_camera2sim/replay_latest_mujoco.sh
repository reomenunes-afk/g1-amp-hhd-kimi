#!/usr/bin/env bash
set -eo pipefail

WORKFLOW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASETS_DIR="$WORKFLOW_DIR/datasets"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Usage: $0 [dataset_name] [--collect] [mujoco args]"
  echo "Replay CameraTrajectory.txt from a dataset in MuJoCo."
  echo "Default: open MuJoCo replay viewer."
  echo "Add --collect to save sim_rgb, primary raycaster depth, and secondary raycaster depth into dataset/mujoco_depth."
  echo "If dataset_name is omitted, use the newest dataset containing CameraTrajectory.txt."
  exit 0
fi

COLLECT_MODE=0
DATASET_ARG=""
FORWARD_ARGS=()
EXPECT_VALUE_FOR=""
ARG_INDEX=0
for arg in "$@"; do
  if [[ -n "$EXPECT_VALUE_FOR" ]]; then
    FORWARD_ARGS+=("$arg")
    EXPECT_VALUE_FOR=""
    ARG_INDEX=$((ARG_INDEX + 1))
    continue
  fi

  if [[ "$arg" == "--collect" ]]; then
    COLLECT_MODE=1
    ARG_INDEX=$((ARG_INDEX + 1))
    continue
  fi

  if [[ "$ARG_INDEX" == "0" && "$arg" != -* ]]; then
    DATASET_ARG="$arg"
    ARG_INDEX=$((ARG_INDEX + 1))
    continue
  fi

  FORWARD_ARGS+=("$arg")

  case "$arg" in
    --xml|--plugin|--body|--camera|--sensor|--output|--every|--max-frames|--render-size|--scale|--z-offset|--basis|--origin|--origin-offset|--depth-block|--min-depth|--max-depth|--depth-clip-min|--depth-clip-max|--render-depth-noise-std|--render-depth-noise-seed|--render-depth-clip-min|--render-depth-clip-max|--viewer-fps)
      EXPECT_VALUE_FOR="$arg"
      ;;
  esac

  ARG_INDEX=$((ARG_INDEX + 1))
done

if [[ -n "$DATASET_ARG" ]]; then
  if [[ "$DATASET_ARG" = /* ]]; then
    DATASET="$DATASET_ARG"
  else
    DATASET="$DATASETS_DIR/$DATASET_ARG"
  fi
else
  DATASET="$(find "$DATASETS_DIR" -mindepth 1 -maxdepth 1 -type d -name 'gemini336l_*' -printf '%T@ %p\n' 2>/dev/null | sort -nr | awk 'NR==1 {print $2}')"
fi

if [[ -z "${DATASET:-}" || ! -f "$DATASET/CameraTrajectory.txt" ]]; then
  if [[ -n "$DATASET_ARG" ]]; then
    echo "Dataset with CameraTrajectory.txt not found: $DATASET" >&2
  else
    echo "No dataset with CameraTrajectory.txt found under $DATASETS_DIR" >&2
  fi
  exit 1
fi

if [[ "$COLLECT_MODE" == "1" ]]; then
  echo "Collecting MuJoCo data from: $DATASET"
  exec "$WORKFLOW_DIR/collect_latest_mujoco_depth.sh" "$DATASET" "${FORWARD_ARGS[@]}"
else
  echo "Replaying MuJoCo trajectory from: $DATASET"
  exec "$WORKFLOW_DIR/replay_trajectory_mujoco.sh" "$DATASET/CameraTrajectory.txt" "${FORWARD_ARGS[@]}"
fi
