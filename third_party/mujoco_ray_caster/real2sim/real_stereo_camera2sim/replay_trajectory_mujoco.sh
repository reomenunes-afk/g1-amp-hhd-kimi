#!/usr/bin/env bash
set -eo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 path_to_trajectory_txt [mujoco replay args]" >&2
  exit 1
fi

WORKFLOW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MJ_CONDA_ENV="${MJ_CONDA_ENV:-mj}"
exec conda run --no-capture-output -n "$MJ_CONDA_ENV" python "$WORKFLOW_DIR/replay_trajectory_mujoco.py" "$@"
