#!/usr/bin/env bash
set -eo pipefail

WORKFLOW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECORDINGS_DIR="$WORKFLOW_DIR/recordings"
DATASETS_DIR="$WORKFLOW_DIR/datasets"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Usage: $0 [dataset_name]"
  echo "First run: export the newest recording and compute ORB-SLAM3 trajectory."
  echo "Later runs: if CameraTrajectory.txt already exists, skip recompute and only visualize trajectory + RGB-D playback."
  echo
  echo "Environment:"
  echo "  FORCE=1       Re-export and rerun ORB-SLAM3 even if trajectory files already exist."
  echo "  NO_VIEWER=1   Do not open the 3D viewer after processing."
  exit 0
fi

LATEST_BAG="$(ls -td "$RECORDINGS_DIR"/gemini336l_* 2>/dev/null | head -n 1 || true)"
if [[ -z "${LATEST_BAG:-}" ]]; then
  echo "No recorded bag found under $RECORDINGS_DIR" >&2
  exit 1
fi

DATASET_NAME="${1:-$(basename "$LATEST_BAG")}"
OUT_DATASET="$DATASETS_DIR/$DATASET_NAME"

mkdir -p "$DATASETS_DIR"

echo "Latest bag: $LATEST_BAG"
echo "Output dataset: $OUT_DATASET"

if [[ -f "$OUT_DATASET/CameraTrajectory.txt" && "${FORCE:-0}" != "1" ]]; then
  echo "Existing trajectory found. Skipping export and ORB-SLAM3."
  echo "Set FORCE=1 to recompute from the latest bag."
else
  "$WORKFLOW_DIR/export_bag_to_tum_rgbd.sh" "$LATEST_BAG" "$OUT_DATASET"
  "$WORKFLOW_DIR/run_orbslam_rgbd.sh" "$OUT_DATASET"
fi

if [[ -f "$OUT_DATASET/CameraTrajectory.txt" ]]; then
  "$WORKFLOW_DIR/visualize_trajectory.sh" "$OUT_DATASET/CameraTrajectory.txt" "$OUT_DATASET/CameraTrajectory.png"
fi

if [[ -f "$OUT_DATASET/KeyFrameTrajectory.txt" ]]; then
  "$WORKFLOW_DIR/visualize_trajectory.sh" "$OUT_DATASET/KeyFrameTrajectory.txt" "$OUT_DATASET/KeyFrameTrajectory.png"
fi

echo
echo "Done."
echo "Trajectory files:"
echo "  $OUT_DATASET/CameraTrajectory.txt"
echo "  $OUT_DATASET/KeyFrameTrajectory.txt"
echo "Trajectory plots:"
echo "  $OUT_DATASET/CameraTrajectory.png"
echo "  $OUT_DATASET/KeyFrameTrajectory.png"

if [[ "${NO_VIEWER:-0}" != "1" && -f "$OUT_DATASET/CameraTrajectory.txt" ]]; then
  echo
  echo "Opening looping 3D viewer with synchronized RGB-D frames. Close the windows to exit this command."
  "$WORKFLOW_DIR/view_trajectory_3d.sh" "$OUT_DATASET/CameraTrajectory.txt" --dataset "$OUT_DATASET"
fi
