#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np


def load_associations(dataset_dir):
    assoc_path = dataset_dir / "associations.txt"
    if not assoc_path.exists():
        raise SystemExit(f"associations.txt not found: {assoc_path}")
    rows = []
    with assoc_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            rgb_t = float(parts[0])
            depth_t = float(parts[2])
            rows.append(
                {
                    "timestamp": rgb_t,
                    "depth_timestamp": depth_t,
                    "rgb": dataset_dir / parts[1],
                    "depth": dataset_dir / parts[3],
                }
            )
    if not rows:
        raise SystemExit(f"No valid rows in: {assoc_path}")
    return rows


def load_sim_frames(sim_dir):
    frames_path = sim_dir / "frames.txt"
    if not frames_path.exists():
        raise SystemExit(f"frames.txt not found: {frames_path}")
    frames = []
    with frames_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            third_depth_npy = None if parts[7] == "-" else sim_dir / parts[7]
            third_depth_png = None if parts[8] == "-" else sim_dir / parts[8]
            third_depth_vis = None if parts[9] == "-" else sim_dir / parts[9]
            frames.append(
                {
                    "out_index": int(parts[0]),
                    "traj_index": int(parts[1]),
                    "timestamp": float(parts[2]),
                    "ray_depth_npy": sim_dir / parts[3],
                    "ray_depth_png": sim_dir / parts[4],
                    "ray_depth_vis": sim_dir / parts[5],
                    "sim_rgb": None if parts[6] == "-" else sim_dir / parts[6],
                    "third_depth_npy": third_depth_npy,
                    "third_depth_png": third_depth_png,
                    "third_depth_vis": third_depth_vis,
                }
            )
    if not frames:
        raise SystemExit(f"No valid rows in: {frames_path}")
    return frames


def load_collection_config(sim_dir):
    config_path = sim_dir / "collection_config.json"
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def nearest_by_timestamp(rows, times, timestamp):
    idx = int(np.argmin(np.abs(times - timestamp)))
    return rows[idx]


def load_real_depth_m(path, depth_factor):
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError(f"Failed to read real depth: {path}")
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    if np.issubdtype(depth.dtype, np.integer):
        return depth.astype(np.float32) / float(depth_factor)
    return depth.astype(np.float32)


def clip_depth_for_range(depth_m, min_depth, max_depth):
    depth = np.asarray(depth_m, dtype=np.float32).copy()
    valid = np.isfinite(depth) & (depth > 0.0)
    depth[~valid] = 0.0
    depth[valid & (depth < min_depth)] = 0.0
    depth[valid & (depth > max_depth)] = max_depth
    return depth


def zero_outside_depth_range(depth_m, min_depth, max_depth):
    depth = np.asarray(depth_m, dtype=np.float32).copy()
    valid = np.isfinite(depth) & (depth > 0.0)
    depth[~valid] = 0.0
    depth[valid & (depth < min_depth)] = 0.0
    depth[valid & (depth > max_depth)] = 0.0
    return depth


def depth_to_vis(depth_m, min_depth, max_depth, gain=1.0):
    depth = clip_depth_for_range(depth_m, min_depth, max_depth)
    valid = np.isfinite(depth) & (depth > 0.0)
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    depth_range = max(max_depth - min_depth, 1e-6)
    normalized[valid] = np.clip((depth[valid] - min_depth) / depth_range * 255.0 * max(gain, 0.0), 0, 255).astype(np.uint8)
    return normalized


def fit_tile(image, width, height):
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)


def put_label(image, text):
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def read_sensor_depth_range(xml_path, sensor_name):
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return None
    for plugin in root.findall(".//sensor/plugin"):
        if plugin.attrib.get("name") != sensor_name:
            continue
        for config in plugin.findall("config"):
            if config.attrib.get("key") != "dis_range":
                continue
            try:
                values = [float(v) for v in config.attrib.get("value", "").replace(",", " ").split()]
            except ValueError:
                return None
            if len(values) >= 2:
                return values[0], values[1]
    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Play real RGB-D and MuJoCo RGB-D comparison frames.")
    parser.add_argument("dataset", help="Dataset directory containing associations.txt and mujoco_depth/")
    parser.add_argument("--sim-dir", help="Override MuJoCo depth output directory. Defaults to DATASET/mujoco_depth.")
    parser.add_argument("--xml", default=str(Path(__file__).with_name("mujoco_camera_replay.xml")), help="MuJoCo XML used to read the default ray caster dis_range.")
    parser.add_argument("--sensor", default="tracked_depth_raycaster", help="Ray caster sensor name in the XML.")
    parser.add_argument("--depth-factor", type=float, default=1000.0, help="Real depth factor; uint16 depth / factor = meters.")
    parser.add_argument("--min-depth", type=float, help="Depth visualization minimum in meters. Defaults to the MJCF sensor dis_range minimum.")
    parser.add_argument("--max-depth", type=float, help="Depth visualization maximum in meters. Defaults to the MJCF sensor dis_range maximum.")
    parser.add_argument("--depth-clip-min", type=float, help="If set together with --depth-clip-max, real_depth and raycastercamera_depth below this become 0.")
    parser.add_argument("--depth-clip-max", type=float, help="If set together with --depth-clip-min, real_depth and raycastercamera_depth above this become 0.")
    parser.add_argument("--depth-vis-gain", type=float, default=1.0, help="Multiply depth visualization brightness by this factor; affects display/export only, not raw depth data.")
    parser.add_argument("--fps", type=float, default=15.0, help="Playback FPS.")
    parser.add_argument("--scale", type=int, default=10, help="Deprecated compatibility option; RGB display keeps real RGB resolution.")
    parser.add_argument("--window-width", type=int, default=960, help="Displayed window width in pixels.")
    parser.add_argument(
        "--save-video",
        nargs="?",
        const="__AUTO__",
        help="Optional path to save the comparison as MP4. If no path is given, save to DATASET/mujoco_depth_comparison.mp4.",
    )
    parser.add_argument("--save-first", help="Save the first comparison frame to this image path and exit.")
    parser.add_argument("--no-gui", action="store_true", help="Do not open an OpenCV window. Use with --save-video or --save-first.")
    parser.add_argument("--max-frames", type=int, default=0, help="Play/export at most this many frames; 0 means all.")
    parser.add_argument("--no-loop", action="store_true", help="Stop at the final frame instead of looping.")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset).expanduser().resolve()
    sim_dir = Path(args.sim_dir).expanduser().resolve() if args.sim_dir else dataset_dir / "mujoco_depth"
    associations = load_associations(dataset_dir)
    frames = load_sim_frames(sim_dir)
    collection_config = load_collection_config(sim_dir)
    assoc_times = np.asarray([row["timestamp"] for row in associations], dtype=np.float64)
    sensor_depth_range = read_sensor_depth_range(Path(args.xml).expanduser().resolve(), args.sensor)
    vis_min_depth = args.min_depth if args.min_depth is not None else collection_config.get("min_depth")
    if vis_min_depth is None:
        vis_min_depth = sensor_depth_range[0] if sensor_depth_range else 0.0
    vis_max_depth = args.max_depth if args.max_depth is not None else collection_config.get("max_depth")
    if vis_max_depth is None:
        vis_max_depth = sensor_depth_range[1] if sensor_depth_range else 2.0
    depth_clip_min = args.depth_clip_min if args.depth_clip_min is not None else collection_config.get("depth_clip_min")
    depth_clip_max = args.depth_clip_max if args.depth_clip_max is not None else collection_config.get("depth_clip_max")
    depth_clip_enabled = depth_clip_min is not None and depth_clip_max is not None
    if (args.depth_clip_min is None) != (args.depth_clip_max is None):
        raise SystemExit("--depth-clip-min and --depth-clip-max must be provided together.")
    if args.max_frames > 0:
        frames = frames[: args.max_frames]

    first_depth = np.load(frames[0]["ray_depth_npy"])
    sim_h, sim_w = first_depth.shape
    first_assoc = nearest_by_timestamp(associations, assoc_times, frames[0]["timestamp"])
    first_rgb = cv2.imread(str(first_assoc["rgb"]), cv2.IMREAD_COLOR)
    if first_rgb is None:
        raise SystemExit(f"Failed to read first real RGB: {first_assoc['rgb']}")
    tile_h, tile_w = first_rgb.shape[:2]
    canvas_size = (tile_w * 3, tile_h * 2)

    writer = None
    save_video_path = None
    if args.save_video:
        if args.save_video == "__AUTO__":
            save_video_path = dataset_dir / "mujoco_depth_comparison.mp4"
        else:
            save_video_path = Path(args.save_video).expanduser()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(save_video_path), fourcc, args.fps, canvas_size)

    print(f"Dataset: {dataset_dir}")
    print(f"Sim frames: {len(frames)} from {sim_dir}")
    print(f"Ray caster size: {sim_w}x{sim_h}")
    print(f"RGB tile size: {tile_w}x{tile_h}")
    print(f"Depth visualization range: {vis_min_depth}..{vis_max_depth} m")
    print(f"Depth visualization gain: {args.depth_vis_gain}")
    if depth_clip_enabled:
        print(f"Depth zero-outside clip: {depth_clip_min}..{depth_clip_max} m")
    if save_video_path is not None:
        print(f"Saving video to: {save_video_path}")
    if len(frames) <= 1:
        print("Warning: only one sim frame was found. Re-run collect_latest_mujoco_depth.sh without --max-frames to get motion.")
    if not args.no_gui:
        print("Keys: q/ESC exit, space pause")

    idx = 0
    paused = False
    delay_ms = max(1, int(1000.0 / max(args.fps, 1.0)))
    window_name = "real/sim RGB-D comparison"
    if not args.no_gui and not args.save_first:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, int(args.window_width), max(1, int(args.window_width * canvas_size[1] / canvas_size[0])))
    try:
        while True:
            frame = frames[idx]
            assoc = nearest_by_timestamp(associations, assoc_times, frame["timestamp"])

            real_rgb = cv2.imread(str(assoc["rgb"]), cv2.IMREAD_COLOR)
            if real_rgb is None:
                raise RuntimeError(f"Failed to read real RGB: {assoc['rgb']}")
            real_depth = load_real_depth_m(assoc["depth"], args.depth_factor)
            ray_depth = np.load(frame["ray_depth_npy"]).astype(np.float32)
            third_depth = None
            if frame["third_depth_npy"] is not None and frame["third_depth_npy"].exists():
                third_depth = np.load(frame["third_depth_npy"]).astype(np.float32)

            real_depth_small = cv2.resize(real_depth, (sim_w, sim_h), interpolation=cv2.INTER_NEAREST)
            if depth_clip_enabled:
                real_depth_small = zero_outside_depth_range(real_depth_small, depth_clip_min, depth_clip_max)
                ray_depth = zero_outside_depth_range(ray_depth, depth_clip_min, depth_clip_max)
                if third_depth is not None:
                    third_depth = zero_outside_depth_range(third_depth, depth_clip_min, depth_clip_max)
            real_depth_small = clip_depth_for_range(real_depth_small, vis_min_depth, vis_max_depth)
            ray_depth = clip_depth_for_range(ray_depth, vis_min_depth, vis_max_depth)
            if third_depth is None:
                third_depth = np.zeros_like(real_depth, dtype=np.float32)
            third_depth = clip_depth_for_range(third_depth, vis_min_depth, vis_max_depth)

            if frame["sim_rgb"] is not None and frame["sim_rgb"].exists():
                sim_rgb = cv2.imread(str(frame["sim_rgb"]), cv2.IMREAD_COLOR)
                if sim_rgb is None:
                    sim_rgb = np.zeros((sim_h, sim_w, 3), dtype=np.uint8)
            else:
                sim_rgb = np.zeros((sim_h, sim_w, 3), dtype=np.uint8)

            tiles = [
                put_label(fit_tile(real_rgb, tile_w, tile_h), "real_rgb"),
                put_label(fit_tile(sim_rgb, tile_w, tile_h), "sim_rgb"),
                put_label(fit_tile(sim_rgb, tile_w, tile_h), "sim_rgb"),
                put_label(fit_tile(depth_to_vis(real_depth_small, vis_min_depth, vis_max_depth, args.depth_vis_gain), tile_w, tile_h), f"real_depth resized to {sim_w}x{sim_h}"),
                put_label(fit_tile(depth_to_vis(ray_depth, vis_min_depth, vis_max_depth, args.depth_vis_gain), tile_w, tile_h), "stereo_raycaster_camera"),
                put_label(fit_tile(depth_to_vis(third_depth, vis_min_depth, vis_max_depth, args.depth_vis_gain), tile_w, tile_h), "raycaster_camera"),
            ]
            top = np.hstack((tiles[0], tiles[1], tiles[2]))
            bottom = np.hstack((tiles[3], tiles[4], tiles[5]))
            canvas = np.vstack((top, bottom))
            cv2.putText(
                canvas,
                f"frame {idx + 1}/{len(frames)}  sim_t={frame['timestamp']:.6f}  real_t={assoc['timestamp']:.6f}",
                (10, canvas.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            if args.save_first:
                out_path = Path(args.save_first).expanduser().resolve()
                out_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_path), canvas)
                print(f"Saved first comparison frame: {out_path}")
                break

            if writer is not None:
                writer.write(canvas)

            if args.no_gui:
                idx += 1
                if idx >= len(frames):
                    break
                continue

            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(delay_ms if not paused else 0) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord(" "):
                paused = not paused
            if not paused:
                idx += 1
                if idx >= len(frames):
                    if args.no_loop:
                        break
                    idx = 0
    finally:
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
