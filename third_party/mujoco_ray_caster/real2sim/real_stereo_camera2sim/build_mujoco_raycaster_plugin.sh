#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAYCASTER_DIR="$(cd "$PROJECT_DIR/../.." && pwd)"
BUILD_DIR="$PROJECT_DIR/build/raycaster_plugin"
OUTPUT_DIR="$RAYCASTER_DIR/lib"
MJ_CONDA_ENV="${MJ_CONDA_ENV:-mj}"

MUJOCO_PY_DIR="$(conda run --no-capture-output -n "$MJ_CONDA_ENV" python -c 'import mujoco, pathlib; print(pathlib.Path(mujoco.__file__).resolve().parent)')"
MUJOCO_INCLUDE_DIR="$MUJOCO_PY_DIR/include"
MUJOCO_LIBRARY="$MUJOCO_PY_DIR/libmujoco.so.3.5.0"

if [[ ! -f "$MUJOCO_INCLUDE_DIR/mujoco/mujoco.h" ]]; then
  echo "MuJoCo header not found under: $MUJOCO_INCLUDE_DIR" >&2
  exit 1
fi
if [[ ! -f "$MUJOCO_LIBRARY" ]]; then
  echo "MuJoCo library not found: $MUJOCO_LIBRARY" >&2
  exit 1
fi

mkdir -p "$BUILD_DIR/shim_include/engine" "$OUTPUT_DIR"
touch \
  "$BUILD_DIR/shim_include/engine/engine_name.h" \
  "$BUILD_DIR/shim_include/engine/engine_util_errmem.h" \
  "$BUILD_DIR/shim_include/engine/engine_util_spatial.h"

cmake -S "$PROJECT_DIR/cmake/raycaster_plugin" -B "$BUILD_DIR" \
  -DRAYCASTER_SOURCE_DIR="$RAYCASTER_DIR" \
  -DMUJOCO_INCLUDE_DIR="$MUJOCO_INCLUDE_DIR" \
  -DMUJOCO_LIBRARY="$MUJOCO_LIBRARY" \
  -DOUTPUT_DIR="$OUTPUT_DIR"
cmake --build "$BUILD_DIR" -j"$(nproc)"

rm -f "$OUTPUT_DIR/libmujoco.so.3.5.0"
cp -f "$MUJOCO_LIBRARY" "$OUTPUT_DIR/libmujoco.so.3.5.0"

echo "Built: $OUTPUT_DIR/libsensor_raycaster.so"
