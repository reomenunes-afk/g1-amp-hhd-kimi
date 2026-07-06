#!/usr/bin/env python3
import argparse
import os
import time
from pathlib import Path

import cv2
import mujoco
import mujoco.viewer
import numpy as np


def load_optional_raycaster_plugin():
    script_dir = Path(__file__).resolve().parent
    candidates = []
    env_path = Path(os.environ["RAYCASTER_PLUGIN"]).expanduser() if "RAYCASTER_PLUGIN" in os.environ else None
    if env_path is not None:
        candidates.append(env_path)
    candidates.extend(
        [
            script_dir / "mujoco_plugins" / "libsensor_raycaster.so",
            script_dir.parents[1] / "lib" / "libsensor_raycaster.so",
        ]
    )
    for plugin_path in candidates:
        if plugin_path.exists():
            mujoco.mj_loadPluginLibrary(str(plugin_path))
            return


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
            timestamp = float(parts[0])
            position = np.asarray([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
            quat_xyzw = np.asarray([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])], dtype=np.float64)
            poses.append((timestamp, position, quat_xyzw))
    if not poses:
        raise SystemExit(f"No valid TUM trajectory poses found in {path}")
    timestamps = np.asarray([p[0] for p in poses], dtype=np.float64)
    positions = np.asarray([p[1] for p in poses], dtype=np.float64)
    quats_xyzw = np.asarray([p[2] for p in poses], dtype=np.float64)
    return timestamps, positions, quats_xyzw


def find_mujoco_camera(model, camera_name):
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if camera_id < 0:
        raise RuntimeError(f"Camera not found: {camera_name}")
    return camera_id


def xyzw_to_wxyz(quat_xyzw):
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float64)
    norm = np.linalg.norm(quat_xyzw)
    if norm < 1e-12:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    x, y, z, w = quat_xyzw / norm
    return np.asarray([w, x, y, z], dtype=np.float64)


def quat_wxyz_to_matrix(quat_wxyz):
    w, x, y, z = [float(v) for v in quat_wxyz]
    norm = np.linalg.norm([w, x, y, z])
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quat_wxyz(rotation):
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, np.asarray(rotation, dtype=np.float64).reshape(-1))
    quat /= max(np.linalg.norm(quat), 1e-12)
    return quat


def parse_vec3(text, name):
    try:
        values = [float(v) for v in text.replace(",", " ").split()]
    except ValueError as exc:
        raise SystemExit(f"Invalid {name}: {text}") from exc
    if len(values) != 3:
        raise SystemExit(f"{name} must contain three numbers, got: {text}")
    return np.asarray(values, dtype=np.float64)


def parse_basis(text):
    presets = {
        # Keep ORB-SLAM3/OpenCV world coordinates as MuJoCo world coordinates.
        "identity": np.eye(3, dtype=np.float64),
        # Common visualization mapping: ORB X right, Y down, Z forward -> MuJoCo X right, Y forward, Z up.
        "opencv_to_mujoco": np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=np.float64),
        # Same idea, but with MuJoCo +Y pointing backward instead of forward.
        "opencv_to_mujoco_y_back": np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64),
        "ros_to_mujoco": np.eye(3, dtype=np.float64),
    }
    if text in presets:
        return presets[text]
    try:
        values = [float(v) for v in text.replace(",", " ").split()]
    except ValueError as exc:
        raise SystemExit(f"Invalid basis: {text}") from exc
    if len(values) != 9:
        raise SystemExit("Basis must be one of identity/opencv_to_mujoco/ros_to_mujoco or 9 row-major numbers.")
    return np.asarray(values, dtype=np.float64).reshape(3, 3)


def transform_poses(positions, quats_wxyz, scale, z_offset, basis, origin_mode, origin_offset):
    transformed_positions = np.asarray(positions, dtype=np.float64).copy() * scale
    transformed_rotations = []

    if origin_mode == "first":
        origin = transformed_positions[0].copy()
    elif origin_mode == "zero":
        origin = np.zeros(3, dtype=np.float64)
    else:
        raise SystemExit(f"Unknown origin mode: {origin_mode}")

    transformed_positions = (basis @ (transformed_positions - origin).T).T + origin_offset
    transformed_positions[:, 2] += z_offset

    for quat in quats_wxyz:
        rotation = quat_wxyz_to_matrix(quat)
        transformed_rotations.append(matrix_to_quat_wxyz(basis @ rotation))

    return transformed_positions, np.asarray(transformed_rotations, dtype=np.float64)


def axis_summary(basis):
    names = ["ORB +X", "ORB +Y", "ORB +Z"]
    labels = ["MuJoCo X", "MuJoCo Y", "MuJoCo Z"]
    rows = []
    for name, vec in zip(names, np.eye(3, dtype=np.float64)):
        mapped = basis @ vec
        axis = int(np.argmax(np.abs(mapped)))
        sign = "+" if mapped[axis] >= 0 else "-"
        rows.append(f"{name}->{sign}{labels[axis]}")
    return ", ".join(rows)


def transform_positions(positions, scale, z_offset):
    transformed = np.asarray(positions, dtype=np.float64).copy()
    transformed *= scale
    transformed[:, 2] += z_offset
    return transformed


def set_mocap_pose(model, data, body_name, position, quat_wxyz):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise RuntimeError(f"Body not found: {body_name}")
    mocap_id = int(model.body_mocapid[body_id])
    if mocap_id < 0:
        raise RuntimeError(f"Body is not mocap: {body_name}")
    data.mocap_pos[mocap_id] = position
    data.mocap_quat[mocap_id] = quat_wxyz
    mujoco.mj_forward(model, data)


def add_path_to_scene(model, positions):
    del model
    # Downsample the trajectory when needed so we do not exceed MuJoCo viewer's
    # user-geometry budget.
    if len(positions) <= 800:
        return positions
    indices = np.linspace(0, len(positions) - 1, 800).astype(int)
    return positions[indices]


def draw_path_geoms(user_scn, positions):
    if user_scn is None or len(positions) < 2:
        return

    user_scn.ngeom = 0
    rgba = np.asarray([0.0, 0.35, 1.0, 1.0], dtype=np.float32)
    size = np.zeros(3, dtype=np.float64)
    pos = np.zeros(3, dtype=np.float64)
    mat = np.eye(3, dtype=np.float64).reshape(-1)

    for p0, p1 in zip(positions[:-1], positions[1:]):
        if user_scn.ngeom >= user_scn.maxgeom:
            break
        geom = user_scn.geoms[user_scn.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE, size, pos, mat, rgba)
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 3.0, p0, p1)
        user_scn.ngeom += 1


def render_camera_frame(model, data, camera_name, width, height):
    renderer = mujoco.Renderer(model, width=width, height=height)
    renderer.update_scene(data, camera=camera_name)
    rgb = renderer.render()
    renderer.close()
    return rgb


def save_render(render_dir, frame_index, image_rgb):
    render_dir.mkdir(parents=True, exist_ok=True)
    out = render_dir / f"{frame_index:06d}.png"
    cv2.imwrite(str(out), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


def render_sequence(model, data, body_name, camera_name, timestamps, positions, quats_wxyz, render_dir, width, height, render_every):
    render_dir.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, width=width, height=height)
    metadata_path = render_dir / "frames.txt"
    count = 0
    try:
        with metadata_path.open("w") as meta:
            meta.write("# frame_index timestamp image\n")
            for idx in range(0, len(positions), max(1, render_every)):
                set_mocap_pose(model, data, body_name, positions[idx], quats_wxyz[idx])
                renderer.update_scene(data, camera=camera_name)
                rgb = renderer.render()
                image_name = f"{idx:06d}.png"
                cv2.imwrite(str(render_dir / image_name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                meta.write(f"{idx} {timestamps[idx]:.9f} {image_name}\n")
                count += 1
    finally:
        renderer.close()
    print(f"Rendered {count} frames to: {render_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Replay an ORB-SLAM3 TUM camera trajectory with a MuJoCo mocap body.")
    parser.add_argument("trajectory", help="Path to CameraTrajectory.txt or KeyFrameTrajectory.txt")
    parser.add_argument("--xml", default=str(Path(__file__).with_name("mujoco_camera_replay.xml")), help="MuJoCo XML scene path.")
    parser.add_argument("--body", default="tracked_camera_mocap", help="Mocap body name in the XML.")
    parser.add_argument("--camera", default="tracked_camera", help="Camera name mounted on the mocap body.")
    parser.add_argument("--hide-mj-camera-vis", action="store_true", help="Disable MuJoCo mjVIS_CAMERA visualization.")
    parser.add_argument("--fps", type=float, default=30.0, help="Playback FPS.")
    parser.add_argument("--step", type=int, default=1, help="Advance this many poses per playback frame.")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale trajectory positions before replay.")
    parser.add_argument("--z-offset", type=float, default=0.0, help="Add this offset to Z after basis mapping.")
    parser.add_argument("--basis", default="opencv_to_mujoco", help="World basis mapping: identity, opencv_to_mujoco, ros_to_mujoco, or 9 row-major numbers.")
    parser.add_argument("--origin", default="first", choices=("first", "zero"), help="Use first trajectory pose or absolute zero as MuJoCo replay origin.")
    parser.add_argument("--origin-offset", default="0 0 0", help="XYZ offset added after origin/basis mapping, e.g. '1 0 0.5'.")
    parser.add_argument("--fixed-view", action="store_true", help="Use fixed overview camera. Default is free interactive view.")
    parser.add_argument("--no-loop", action="store_true", help="Stop at the last pose instead of looping.")
    parser.add_argument("--render-dir", help="Optional directory to save RGB images rendered from the mounted MuJoCo camera.")
    parser.add_argument("--render-size", default="640x480", help="Render size WIDTHxHEIGHT for --render-dir.")
    parser.add_argument("--render-every", type=int, default=1, help="Save one rendered frame every N playback updates.")
    parser.add_argument("--render-only", action="store_true", help="Render the mounted camera along the whole trajectory and exit without opening the viewer.")
    parser.add_argument("--dry-run", action="store_true", help="Load XML and trajectory, set the first mocap pose, then exit.")
    return parser.parse_args()


def main():
    args = parse_args()
    traj_path = Path(args.trajectory).expanduser().resolve()
    xml_path = Path(args.xml).expanduser().resolve()
    timestamps, positions_raw, quats_xyzw = load_tum(traj_path)
    raw_quats_wxyz = np.asarray([xyzw_to_wxyz(q) for q in quats_xyzw], dtype=np.float64)
    basis = parse_basis(args.basis)
    origin_offset = parse_vec3(args.origin_offset, "origin-offset")
    positions, quats_wxyz = transform_poses(
        positions_raw,
        raw_quats_wxyz,
        args.scale,
        args.z_offset,
        basis,
        args.origin,
        origin_offset,
    )

    load_optional_raycaster_plugin()
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    camera_id = find_mujoco_camera(model, args.camera)
    set_mocap_pose(model, data, args.body, positions[0], quats_wxyz[0])
    path_positions = add_path_to_scene(model, positions)

    render_dir = Path(args.render_dir).expanduser().resolve() if args.render_dir else None
    render_width, render_height = [int(v) for v in args.render_size.lower().split("x", 1)]
    render_counter = 0

    print(f"Loaded trajectory: {traj_path}")
    print(f"Poses: {len(positions)}")
    print(f"Scene XML: {xml_path}")
    print(f"MuJoCo camera: {args.camera}, id={camera_id}")
    print(f"  resolution: {model.cam_resolution[camera_id].tolist()}")
    print(f"  sensorsize: {model.cam_sensorsize[camera_id].tolist()}")
    print(f"  intrinsic [focal_x focal_y principal_x principal_y]: {model.cam_intrinsic[camera_id].tolist()}")
    print(f"Basis mapping: {args.basis}")
    print(f"Basis matrix:\n{basis}")
    print(f"Axis mapping: {axis_summary(basis)}")
    print(f"Origin mode: {args.origin}, origin offset: {origin_offset.tolist()}, z offset: {args.z_offset}")
    print(f"Viewer path segments: {max(0, len(path_positions) - 1)}")
    print("Close the MuJoCo viewer window to exit. Default view is free/interactive.")

    if args.dry_run:
        print("Dry run OK.")
        return

    if args.render_only:
        if render_dir is None:
            raise SystemExit("--render-only requires --render-dir")
        render_sequence(
            model,
            data,
            args.body,
            args.camera,
            timestamps,
            positions,
            quats_wxyz,
            render_dir,
            render_width,
            render_height,
            args.render_every,
        )
        return

    idx = 0
    delay = 1.0 / max(args.fps, 1.0)
    last_wall = time.time()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        if not args.hide_mj_camera_vis:
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CAMERA] = 1
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_STATIC] = 1
        if args.fixed_view:
            try:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overview")
            except Exception:
                pass
        else:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.lookat[:] = positions.mean(axis=0)
            viewer.cam.distance = max(1.0, float(np.linalg.norm(positions.max(axis=0) - positions.min(axis=0))) * 1.8)
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -25
        draw_path_geoms(viewer.user_scn, path_positions)

        while viewer.is_running():
            idx += max(1, args.step)
            if idx >= len(positions):
                if args.no_loop:
                    idx = len(positions) - 1
                else:
                    idx = 0

            set_mocap_pose(model, data, args.body, positions[idx], quats_wxyz[idx])

            if render_dir and render_counter % max(1, args.render_every) == 0:
                image = render_camera_frame(model, data, args.camera, render_width, render_height)
                save_render(render_dir, idx, image)
            render_counter += 1

            viewer.sync()

            # Pace playback using wall time. If rendering is slow, do not try to catch up aggressively.
            now = time.time()
            sleep_for = delay - (now - last_wall)
            if sleep_for > 0:
                time.sleep(sleep_for)
            last_wall = time.time()

            if args.no_loop and idx == len(positions) - 1:
                time.sleep(0.5)


if __name__ == "__main__":
    main()
