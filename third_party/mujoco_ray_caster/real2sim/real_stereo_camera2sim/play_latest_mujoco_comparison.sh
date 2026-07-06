#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASETS_DIR="$PROJECT_DIR/datasets"
MJ_CONDA_ENV="${MJ_CONDA_ENV:-mj}"

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

exec conda run --no-capture-output -n "$MJ_CONDA_ENV" python "$PROJECT_DIR/play_mujoco_depth_comparison.py" \
  "$DATASET_DIR" \
  "$@"
