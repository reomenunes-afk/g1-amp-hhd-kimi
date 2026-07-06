#!/usr/bin/env python3
import argparse
import bisect
import math
import os
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.serialization import deserialize_message
import rosbag2_py
from rosidl_runtime_py.utilities import get_message


def stamp_to_sec(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def msg_time(msg, bag_time_ns):
    if hasattr(msg, "header"):
        t = stamp_to_sec(msg.header.stamp)
        if t > 0:
            return t
    return float(bag_time_ns) * 1e-9


def row_view(msg, dtype, channels):
    height = int(msg.height)
    width = int(msg.width)
    step = int(msg.step)
    data = np.frombuffer(msg.data, dtype=dtype)
    row_elems = step // np.dtype(dtype).itemsize
    arr = data.reshape((height, row_elems))
    if channels == 1:
        return arr[:, :width].copy()
    return arr[:, : width * channels].reshape((height, width, channels)).copy()


def image_to_bgr(msg):
    enc = msg.encoding.lower()
    if enc in ("bgr8", "8uc3"):
        return row_view(msg, np.uint8, 3)
    if enc == "rgb8":
        return cv2.cvtColor(row_view(msg, np.uint8, 3), cv2.COLOR_RGB2BGR)
    if enc == "bgra8":
        return cv2.cvtColor(row_view(msg, np.uint8, 4), cv2.COLOR_BGRA2BGR)
    if enc == "rgba8":
        return cv2.cvtColor(row_view(msg, np.uint8, 4), cv2.COLOR_RGBA2BGR)
    if enc in ("mono8", "8uc1"):
        return row_view(msg, np.uint8, 1)
    if enc in ("yuyv", "yuyv422", "yuv422"):
        yuyv = row_view(msg, np.uint8, 2)
        return cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUY2)
    if enc == "uyvy":
        uyvy = row_view(msg, np.uint8, 2)
        return cv2.cvtColor(uyvy, cv2.COLOR_YUV2BGR_UYVY)
    raise ValueError(f"Unsupported RGB encoding: {msg.encoding}")


def depth_to_uint16_mm(msg):
    enc = msg.encoding.lower()
    if enc in ("16uc1", "mono16"):
        depth = row_view(msg, np.uint16, 1)
        return depth
    if enc == "32fc1":
        depth_m = row_view(msg, np.float32, 1)
        depth_mm = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(depth_mm * 1000.0, 0, 65535).astype(np.uint16)
    raise ValueError(f"Unsupported depth encoding: {msg.encoding}")


def open_bag_reader(bag_path):
    storage_options = rosbag2_py.StorageOptions(uri=str(bag_path), storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    return reader


def write_settings(path, camera_info, fps, depth_factor, rgb_flag):
    k = list(camera_info.k)
    d = list(camera_info.d)
    d += [0.0] * max(0, 5 - len(d))
    width = int(camera_info.width)
    height = int(camera_info.height)

    fps_int = max(1, int(round(fps)))

    text = f"""%YAML:1.0

File.version: "1.0"
Camera.type: "PinHole"

Camera1.fx: {k[0]:.10f}
Camera1.fy: {k[4]:.10f}
Camera1.cx: {k[2]:.10f}
Camera1.cy: {k[5]:.10f}

Camera1.k1: {d[0]:.10f}
Camera1.k2: {d[1]:.10f}
Camera1.p1: {d[2]:.10f}
Camera1.p2: {d[3]:.10f}
Camera1.k3: {d[4]:.10f}

Camera.width: {width}
Camera.height: {height}
Camera.fps: {fps_int}
Camera.RGB: {rgb_flag}

Stereo.ThDepth: 40.0
Stereo.b: 0.07732

RGBD.DepthMapFactor: {depth_factor:.6f}

ORBextractor.nFeatures: 1000
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7

Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500.0
"""
    path.write_text(text)


def median_fps(timestamps):
    if len(timestamps) < 2:
        return 30.0
    deltas = [b - a for a, b in zip(timestamps[:-1], timestamps[1:]) if b > a]
    if not deltas:
        return 30.0
    median_dt = float(np.median(deltas))
    if median_dt <= 0:
        return 30.0
    return 1.0 / median_dt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a ROS2 Orbbec RGB-D bag to ORB-SLAM3 TUM RGB-D format."
    )
    parser.add_argument("bag", help="Path to rosbag2 directory.")
    parser.add_argument("output", help="Output dataset directory.")
    parser.add_argument("--rgb-topic", default="/camera/color/image_raw")
    parser.add_argument("--depth-topic", default="/camera/depth/image_raw")
    parser.add_argument("--camera-info-topic", default="/camera/color/camera_info")
    parser.add_argument("--max-dt", type=float, default=0.03, help="Max RGB/depth timestamp gap in seconds.")
    parser.add_argument("--depth-factor", type=float, default=1000.0, help="ORB-SLAM3 depth factor for saved PNG depth.")
    parser.add_argument("--rgb-flag", type=int, default=1, choices=(0, 1), help="Camera.RGB value written to YAML.")
    return parser.parse_args()


def main():
    args = parse_args()
    bag_path = Path(args.bag).expanduser().resolve()
    out_dir = Path(args.output).expanduser().resolve()
    rgb_dir = out_dir / "rgb"
    depth_dir = out_dir / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    reader = open_bag_reader(bag_path)
    topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    needed = [args.rgb_topic, args.depth_topic, args.camera_info_topic]
    missing = [topic for topic in needed if topic not in topic_types]
    if missing:
        raise SystemExit(f"Missing topics in bag: {', '.join(missing)}")

    type_cache = {topic: get_message(type_name) for topic, type_name in topic_types.items()}
    rgb_entries = []
    depth_entries = []
    camera_info = None
    rgb_count = 0
    depth_count = 0

    while reader.has_next():
        topic, data, bag_time = reader.read_next()
        if topic not in (args.rgb_topic, args.depth_topic, args.camera_info_topic):
            continue

        msg = deserialize_message(data, type_cache[topic])

        if topic == args.camera_info_topic and camera_info is None:
            camera_info = msg
            continue

        t = msg_time(msg, bag_time)
        if topic == args.rgb_topic:
            image = image_to_bgr(msg)
            rel = f"rgb/{rgb_count:06d}.png"
            if not cv2.imwrite(str(out_dir / rel), image):
                raise RuntimeError(f"Failed writing {rel}")
            rgb_entries.append((t, rel))
            rgb_count += 1
        elif topic == args.depth_topic:
            depth = depth_to_uint16_mm(msg)
            rel = f"depth/{depth_count:06d}.png"
            if not cv2.imwrite(str(out_dir / rel), depth):
                raise RuntimeError(f"Failed writing {rel}")
            depth_entries.append((t, rel))
            depth_count += 1

    if camera_info is None:
        raise SystemExit(f"No CameraInfo received on {args.camera_info_topic}")
    if not rgb_entries or not depth_entries:
        raise SystemExit("Bag did not contain both RGB and depth frames.")

    rgb_entries.sort(key=lambda item: item[0])
    depth_entries.sort(key=lambda item: item[0])
    depth_times = [x[0] for x in depth_entries]
    associations = []
    used_depth = set()
    for rgb_t, rgb_rel in rgb_entries:
        pos = bisect.bisect_left(depth_times, rgb_t)
        candidates = []
        if pos < len(depth_entries):
            candidates.append(pos)
        if pos > 0:
            candidates.append(pos - 1)
        best = None
        best_dt = math.inf
        for idx in candidates:
            if idx in used_depth:
                continue
            dt = abs(depth_entries[idx][0] - rgb_t)
            if dt < best_dt:
                best = idx
                best_dt = dt
        if best is not None and best_dt <= args.max_dt:
            used_depth.add(best)
            depth_t, depth_rel = depth_entries[best]
            associations.append((rgb_t, rgb_rel, depth_t, depth_rel))

    if not associations:
        raise SystemExit(
            f"No RGB/depth pairs found within --max-dt={args.max_dt}. "
            "Try a larger value or check synchronization."
        )

    assoc_path = out_dir / "associations.txt"
    with assoc_path.open("w") as f:
        for rgb_t, rgb_rel, depth_t, depth_rel in associations:
            f.write(f"{rgb_t:.9f} {rgb_rel} {depth_t:.9f} {depth_rel}\n")

    fps = median_fps([x[0] for x in associations])
    settings_path = out_dir / "orbslam3_gemini336l.yaml"
    write_settings(settings_path, camera_info, fps, args.depth_factor, args.rgb_flag)

    print(f"Exported RGB frames: {len(rgb_entries)}")
    print(f"Exported depth frames: {len(depth_entries)}")
    print(f"Associated pairs: {len(associations)}")
    print(f"Dataset: {out_dir}")
    print(f"Associations: {assoc_path}")
    print(f"ORB-SLAM3 settings: {settings_path}")


if __name__ == "__main__":
    rclpy.init(args=None)
    try:
        main()
    finally:
        rclpy.shutdown()
