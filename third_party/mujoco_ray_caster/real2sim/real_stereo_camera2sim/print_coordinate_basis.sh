#!/usr/bin/env bash
set -eo pipefail

WORKFLOW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MJ_CONDA_ENV="${MJ_CONDA_ENV:-mj}"
exec conda run --no-capture-output -n "$MJ_CONDA_ENV" python "$WORKFLOW_DIR/print_coordinate_basis.py" "$@"
