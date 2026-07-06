#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


def load_tum(path):
    poses = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            t = float(parts[0])
            xyz = [float(parts[1]), float(parts[2]), float(parts[3])]
            qxyzw = [float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])]
            poses.append((t, xyz, qxyzw))
    if not poses:
        raise SystemExit(f"No valid TUM poses found in {path}")
    timestamps = np.asarray([p[0] for p in poses], dtype=np.float64)
    points = np.asarray([p[1] for p in poses], dtype=np.float64)
    quats = np.asarray([p[2] for p in poses], dtype=np.float64)
    return timestamps, points, quats


def make_line_set(points, color):
    lines = [[i, i + 1] for i in range(max(0, len(points) - 1))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color for _ in lines])
    return line_set


def make_points(points, color):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector([color for _ in range(len(points))])
    return cloud


def make_camera_marker(points):
    span = trajectory_span(points)
    axis_len = max(span * 0.08, 0.05)
    frustum_z = max(span * 0.10, 0.08)
    frustum_w = frustum_z * 0.55
    frustum_h = frustum_z * 0.38

    local_points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [axis_len, 0.0, 0.0],
            [0.0, axis_len, 0.0],
            [0.0, 0.0, axis_len],
            [-frustum_w, -frustum_h, frustum_z],
            [frustum_w, -frustum_h, frustum_z],
            [frustum_w, frustum_h, frustum_z],
            [-frustum_w, frustum_h, frustum_z],
        ],
        dtype=np.float64,
    )
    lines = [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [0, 5],
        [0, 6],
        [0, 7],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
    ]
    colors = [
        [1.0, 0.05, 0.05],
        [0.05, 0.75, 0.05],
        [0.05, 0.25, 1.0],
        [0.35, 0.35, 0.35],
        [0.35, 0.35, 0.35],
        [0.35, 0.35, 0.35],
        [0.35, 0.35, 0.35],
        [1.0, 0.55, 0.05],
        [1.0, 0.55, 0.05],
        [1.0, 0.55, 0.05],
        [1.0, 0.55, 0.05],
    ]
    marker = o3d.geometry.LineSet()
    marker.points = o3d.utility.Vector3dVector(local_points)
    marker.lines = o3d.utility.Vector2iVector(lines)
    marker.colors = o3d.utility.Vector3dVector(colors)
    return marker, local_points


def trajectory_span(points):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    return max(float(np.linalg.norm(maxs - mins)), 1e-3)


def quat_xyzw_to_rotation(q):
    x, y, z, w = [float(v) for v in q]
    norm = np.linalg.norm([x, y, z, w])
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def transform_marker_points(local_points, position, quat_xyzw):
    rotation = quat_xyzw_to_rotation(quat_xyzw)
    return local_points @ rotation.T + position


def center_view(vis, points):
    ctr = vis.get_view_control()
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    span = max(float(np.linalg.norm(maxs - mins)), 1e-3)
    ctr.set_lookat(center)
    ctr.set_zoom(0.65 if span < 2.0 else 0.45)
    ctr.set_front([0.6, -0.7, -0.35])
    ctr.set_up([0.0, 0.0, 1.0])


def load_associations(path):
    entries = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            entries.append((float(parts[0]), parts[1], float(parts[2]), parts[3]))
    if not entries:
        raise SystemExit(f"No RGB-D associations found in {path}")
    entries.sort(key=lambda item: item[0])
    return entries


def nearest_association(entries, timestamp):
    times = np.asarray([item[0] for item in entries], dtype=np.float64)
    pos = int(np.searchsorted(times, timestamp))
    candidates = []
    if pos < len(entries):
        candidates.append(pos)
    if pos > 0:
        candidates.append(pos - 1)
    if not candidates:
        return entries[0]
    best = min(candidates, key=lambda idx: abs(entries[idx][0] - timestamp))
    return entries[best]


def colorize_depth(depth):
    if depth is None:
        return None
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    valid = depth[depth > 0]
    if valid.size == 0:
        normalized = np.zeros(depth.shape, dtype=np.uint8)
    else:
        lo = float(np.percentile(valid, 2))
        hi = float(np.percentile(valid, 98))
        if hi <= lo:
            hi = lo + 1.0
        clipped = np.clip(depth.astype(np.float32), lo, hi)
        normalized = ((clipped - lo) * 255.0 / (hi - lo)).astype(np.uint8)
        normalized[depth == 0] = 0
    return cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)


def resize_to_height(image, height):
    if image is None:
        return None
    if image.shape[0] == height:
        return image
    scale = height / float(image.shape[0])
    width = max(1, int(round(image.shape[1] * scale)))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def make_rgbd_panel(dataset, association, trajectory_index, trajectory_len, max_height):
    rgb_t, rgb_rel, depth_t, depth_rel = association
    rgb_path = dataset / rgb_rel
    depth_path = dataset / depth_rel

    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if rgb is None:
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(rgb, f"Missing RGB: {rgb_rel}", (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    depth_vis = colorize_depth(depth)
    if depth_vis is None:
        depth_vis = np.zeros_like(rgb)
        cv2.putText(depth_vis, f"Missing depth: {depth_rel}", (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    target_h = min(max_height, max(rgb.shape[0], depth_vis.shape[0]))
    rgb = resize_to_height(rgb, target_h)
    depth_vis = resize_to_height(depth_vis, target_h)

    label_h = 42
    rgb_header = np.full((label_h, rgb.shape[1], 3), 245, dtype=np.uint8)
    depth_header = np.full((label_h, depth_vis.shape[1], 3), 245, dtype=np.uint8)
    cv2.putText(rgb_header, f"RGB  t={rgb_t:.3f}s", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(depth_header, f"Depth  t={depth_t:.3f}s", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
    rgb = np.vstack([rgb_header, rgb])
    depth_vis = np.vstack([depth_header, depth_vis])

    if rgb.shape[0] != depth_vis.shape[0]:
        h = max(rgb.shape[0], depth_vis.shape[0])
        rgb = resize_to_height(rgb, h)
        depth_vis = resize_to_height(depth_vis, h)

    gap = np.full((rgb.shape[0], 12, 3), 255, dtype=np.uint8)
    panel = np.hstack([rgb, gap, depth_vis])
    footer = np.full((44, panel.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(
        footer,
        f"pose {trajectory_index + 1}/{trajectory_len}    close this image window or press q to exit all playback",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    return np.vstack([panel, footer])


def parse_args():
    parser = argparse.ArgumentParser(description="Loop-play a TUM trajectory with optional synchronized RGB-D frames.")
    parser.add_argument("trajectory", help="CameraTrajectory.txt or KeyFrameTrajectory.txt")
    parser.add_argument("--fps", type=float, default=30.0, help="Playback frames per second.")
    parser.add_argument("--step", type=int, default=1, help="Advance this many trajectory poses per frame.")
    parser.add_argument("--no-loop", action="store_true", help="Stop at the final pose instead of looping.")
    parser.add_argument("--title", default="ORB-SLAM3 RGB-D Trajectory")
    parser.add_argument("--dataset", help="Dataset directory containing rgb/, depth/, and associations.txt.")
    parser.add_argument("--associations", help="Associations file. Defaults to DATASET/associations.txt.")
    parser.add_argument("--no-images", action="store_true", help="Only show 3D trajectory, no RGB-D image window.")
    parser.add_argument("--image-height", type=int, default=480, help="Maximum RGB/depth image height.")
    return parser.parse_args()


def main():
    args = parse_args()
    traj_path = Path(args.trajectory).expanduser().resolve()
    timestamps, points, quats = load_tum(traj_path)
    dataset = Path(args.dataset).expanduser().resolve() if args.dataset else None
    associations = None
    if dataset and not args.no_images:
        associations_path = Path(args.associations).expanduser().resolve() if args.associations else dataset / "associations.txt"
        if associations_path.exists():
            associations = load_associations(associations_path)
        else:
            print(f"Associations not found, RGB-D image window disabled: {associations_path}")

    if len(points) == 1:
        points = np.vstack([points, points + np.array([[1e-3, 0.0, 0.0]])])
        quats = np.vstack([quats, quats[0]])
        timestamps = np.append(timestamps, timestamps[0])

    full_path = make_line_set(points, [0.05, 0.35, 0.95])
    keypoints = make_points(points[:: max(1, len(points) // 250)], [0.05, 0.05, 0.05])
    marker, marker_local_points = make_camera_marker(points)
    marker.points = o3d.utility.Vector3dVector(transform_marker_points(marker_local_points, points[0], quats[0]))
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=max(trajectory_span(points) * 0.08, 0.05))

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=args.title, width=1280, height=800)
    vis.add_geometry(full_path)
    vis.add_geometry(keypoints)
    vis.add_geometry(marker)
    vis.add_geometry(frame)

    render = vis.get_render_option()
    render.background_color = np.asarray([1.0, 1.0, 1.0])
    render.line_width = 4.0
    render.point_size = 4.0
    center_view(vis, points)

    idx = 0
    delay = 1.0 / max(args.fps, 1.0)

    if associations is not None:
        cv2.namedWindow("ORB-SLAM3 RGB-D frames", cv2.WINDOW_NORMAL)
        print("Synchronized RGB-D image window is enabled.")
    print("3D trajectory viewer is running with position and orientation. Close the Open3D window to exit.")
    while True:
        if not vis.poll_events():
            break
        vis.update_renderer()

        idx += max(1, args.step)
        if idx >= len(points):
            if args.no_loop:
                idx = len(points) - 1
            else:
                idx = 0

        new_pos = points[idx]
        marker.points = o3d.utility.Vector3dVector(transform_marker_points(marker_local_points, new_pos, quats[idx]))
        vis.update_geometry(marker)

        if associations is not None:
            assoc = nearest_association(associations, timestamps[idx])
            panel = make_rgbd_panel(dataset, assoc, idx, len(points), args.image_height)
            cv2.imshow("ORB-SLAM3 RGB-D frames", panel)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
        time.sleep(delay)

    vis.destroy_window()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
