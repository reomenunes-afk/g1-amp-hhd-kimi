#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAYCASTER_ROOT="$(cd "$PROJECT_DIR/../.." && pwd)"
BUILD_DIR="$PROJECT_DIR/build/mujoco_rgbd_collector"
OUTPUT_DIR="$PROJECT_DIR/bin"
RAYCASTER_DIR="$RAYCASTER_ROOT"
COLLECTOR_SOURCE="$PROJECT_DIR/src/mujoco_collect_rgbd.cpp"
MJ_CONDA_ENV="${MJ_CONDA_ENV:-mj}"

if [[ ! -d "$RAYCASTER_DIR/raycaster_src" ]]; then
  echo "Missing ray caster source: $RAYCASTER_DIR" >&2
  exit 1
fi

MUJOCO_PY_DIR="$(conda run --no-capture-output -n "$MJ_CONDA_ENV" python -c 'import mujoco, pathlib; print(pathlib.Path(mujoco.__file__).resolve().parent)')"
MUJOCO_INCLUDE_DIR="$MUJOCO_PY_DIR/include"
MUJOCO_LIBRARY="$MUJOCO_PY_DIR/libmujoco.so.3.5.0"
MUJOCO_LIBRARY_DIR="$MUJOCO_PY_DIR"

if [[ ! -f "$MUJOCO_INCLUDE_DIR/mujoco/mujoco.h" ]]; then
  echo "MuJoCo headers not found under: $MUJOCO_INCLUDE_DIR" >&2
  exit 1
fi
if [[ ! -f "$MUJOCO_LIBRARY" ]]; then
  echo "MuJoCo library not found: $MUJOCO_LIBRARY" >&2
  exit 1
fi

mkdir -p "$BUILD_DIR" "$OUTPUT_DIR"

cmake -S "$PROJECT_DIR/cmake/mujoco_rgbd_collector" -B "$BUILD_DIR" \
  -DRAYCASTER_SOURCE_DIR="$RAYCASTER_DIR" \
  -DMUJOCO_INCLUDE_DIR="$MUJOCO_INCLUDE_DIR" \
  -DMUJOCO_LIBRARY="$MUJOCO_LIBRARY" \
  -DMUJOCO_LIBRARY_DIR="$MUJOCO_LIBRARY_DIR" \
  -DCOLLECTOR_SOURCE="$COLLECTOR_SOURCE" \
  -DOUTPUT_DIR="$OUTPUT_DIR"

cmake --build "$BUILD_DIR" -j"$(nproc)"

echo "Built: $OUTPUT_DIR/mujoco_rgbd_collector"
