#!/usr/bin/env bash
set -eo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 path_to_trajectory_txt [output_png]" >&2
  exit 1
fi

WORKFLOW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAJECTORY="$1"
OUTPUT="${2:-}"

if [[ -n "$OUTPUT" ]]; then
  exec python3 "$WORKFLOW_DIR/visualize_trajectory.py" "$TRAJECTORY" --output "$OUTPUT"
else
  exec python3 "$WORKFLOW_DIR/visualize_trajectory.py" "$TRAJECTORY"
fi
