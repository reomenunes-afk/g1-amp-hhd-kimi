#!/usr/bin/env bash
set -eo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 path_to_trajectory_txt [viewer args]" >&2
  exit 1
fi

WORKFLOW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$WORKFLOW_DIR/view_trajectory_3d.py" "$@"
