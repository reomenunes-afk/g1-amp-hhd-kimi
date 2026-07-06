#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import numpy as np


def load_tum(path):
    pts = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            pts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not pts:
        raise SystemExit(f"No valid trajectory points found in {path}")
    return np.asarray(pts, dtype=np.float64)


def project(points_2d, size=900, margin=60):
    mins = points_2d.min(axis=0)
    maxs = points_2d.max(axis=0)
    spans = np.maximum(maxs - mins, 1e-9)
    usable = size - 2 * margin
    scale = usable / max(spans[0], spans[1])
    mapped = (points_2d - mins) * scale
    mapped[:, 0] += margin
    mapped[:, 1] += margin
    mapped[:, 1] = size - mapped[:, 1]
    return mapped.astype(np.int32), mins, maxs


def draw_view(canvas, points_2d, title, axis_labels):
    h, w = canvas.shape[:2]
    cv2.rectangle(canvas, (40, 40), (w - 40, h - 40), (220, 220, 220), 1)
    for i in range(1, len(points_2d)):
        cv2.line(canvas, tuple(points_2d[i - 1]), tuple(points_2d[i]), (50, 170, 255), 2, cv2.LINE_AA)
    cv2.circle(canvas, tuple(points_2d[0]), 5, (60, 220, 60), -1, cv2.LINE_AA)
    cv2.circle(canvas, tuple(points_2d[-1]), 5, (60, 60, 230), -1, cv2.LINE_AA)
    cv2.putText(canvas, title, (50, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"start", (points_2d[0][0] + 8, points_2d[0][1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 120, 30), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"end", (points_2d[-1][0] + 8, points_2d[-1][1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 150), 1, cv2.LINE_AA)
    cv2.putText(canvas, axis_labels[0], (w - 90, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1, cv2.LINE_AA)
    cv2.putText(canvas, axis_labels[1], (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1, cv2.LINE_AA)


def trajectory_length(points):
    if len(points) < 2:
        return 0.0
    diffs = np.diff(points, axis=0)
    return float(np.linalg.norm(diffs, axis=1).sum())


def main():
    parser = argparse.ArgumentParser(description="Visualize a TUM trajectory file as XY/XZ projections.")
    parser.add_argument("trajectory", help="Path to CameraTrajectory.txt or KeyFrameTrajectory.txt")
    parser.add_argument("--output", help="Output PNG path")
    parser.add_argument("--size", type=int, default=900, help="Per-view canvas size in pixels")
    args = parser.parse_args()

    traj_path = Path(args.trajectory).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve() if args.output else traj_path.with_suffix(".png")

    pts = load_tum(traj_path)
    xy, _, _ = project(pts[:, [0, 1]], size=args.size)
    xz, _, _ = project(pts[:, [0, 2]], size=args.size)

    panel_xy = np.full((args.size, args.size, 3), 250, dtype=np.uint8)
    panel_xz = np.full((args.size, args.size, 3), 250, dtype=np.uint8)
    draw_view(panel_xy, xy, "XY View", ("X", "Y"))
    draw_view(panel_xz, xz, "XZ View", ("X", "Z"))

    gap = np.full((args.size, 40, 3), 255, dtype=np.uint8)
    image = np.hstack([panel_xy, gap, panel_xz])

    info = np.full((120, image.shape[1], 3), 255, dtype=np.uint8)
    total_len = trajectory_length(pts)
    xyz_min = pts.min(axis=0)
    xyz_max = pts.max(axis=0)
    lines = [
        f"file: {traj_path.name}",
        f"points: {len(pts)}   path_length: {total_len:.3f} m",
        f"x:[{xyz_min[0]:.3f}, {xyz_max[0]:.3f}] y:[{xyz_min[1]:.3f}, {xyz_max[1]:.3f}] z:[{xyz_min[2]:.3f}, {xyz_max[2]:.3f}]",
    ]
    for i, text in enumerate(lines):
        cv2.putText(info, text, (20, 35 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)

    final = np.vstack([info, image])
    cv2.imwrite(str(out_path), final)
    print(f"Saved trajectory plot to: {out_path}")


if __name__ == "__main__":
    main()
