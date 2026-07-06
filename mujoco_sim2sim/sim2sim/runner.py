"""Standalone MuJoCo runner for the exported G1 parkour policy."""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    import cv2
except ImportError:  # pragma: no cover - runtime dependency.
    cv2 = None

from .depth import DepthConfig, ParkourDepthProcessor
from .math_utils import pd_torque
from .observations import ParkourObservationBuilder
from .policy import ParkourOnnxPolicy
from .robot import G1RobotAdapter


@dataclass
class RunnerConfig:
    path: Path
    data: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "RunnerConfig":
        cfg_path = Path(path).resolve()
        with cfg_path.open("r", encoding="utf-8") as f:
            return cls(path=cfg_path, data=yaml.safe_load(f))

    def resolve(self, key: str) -> Path:
        cur: Any = self.data
        for part in key.split("."):
            cur = cur[part]
        p = Path(cur)
        if not p.is_absolute():
            p = (self.path.parent / p).resolve()
        return p


class MujocoSim2SimRunner:
    def __init__(self, cfg: RunnerConfig):
        import mujoco

        self.cfg = cfg
        self.mujoco = mujoco
        plugin_path = cfg.resolve("paths.raycaster_plugin")
        if plugin_path.exists():
            mujoco.mj_loadPluginLibrary(str(plugin_path))
        else:
            raise FileNotFoundError(f"MuJoCo raycaster plugin not found: {plugin_path}")
        self.model = mujoco.MjModel.from_xml_path(str(cfg.resolve("paths.xml")))
        self.data = mujoco.MjData(self.model)

        sim_cfg = cfg.data["simulation"]
        self.model.opt.timestep = float(sim_cfg["dt"])
        self.decimation = int(sim_cfg["control_decimation"])
        self.policy_dt = self.model.opt.timestep * self.decimation
        self.duration = float(sim_cfg["duration"])
        self.show_depth = bool(sim_cfg.get("show_depth", False))
        self.record_video = bool(sim_cfg.get("record_video", False))
        self.video_path = cfg.resolve("simulation.video_path")
        self.video_width = int(sim_cfg.get("video_width", 1280))
        self.video_height = int(sim_cfg.get("video_height", 720))
        self.video_fps = int(sim_cfg.get("video_fps", 50))
        self.video_camera_cfg = sim_cfg.get("video_camera", {})

        robot_cfg = cfg.data["robot"]
        self.robot = G1RobotAdapter.from_mujoco(mujoco, self.model, robot_cfg["default_joint_pos"])
        self.command = np.array(cfg.data["command"]["default"], dtype=np.float32)
        self.last_action = np.zeros(29, dtype=np.float32)
        self.target_pos = self.robot.default_pos.copy()
        self.imu_yaw_correction_rad = float(robot_cfg.get("imu_yaw_correction_rad", 0.0))

        policy_cfg = cfg.data["policy"]
        self.obs_builder = ParkourObservationBuilder(history_len=int(policy_cfg["proprio_history"]))
        self.policy = ParkourOnnxPolicy(cfg.resolve("paths.actor_onnx"), cfg.resolve("paths.depth_encoder_onnx"))

        depth_cfg_raw = cfg.data["depth"]
        self.depth = ParkourDepthProcessor(
            mujoco,
            self.model,
            DepthConfig(
                camera_name=depth_cfg_raw["camera_name"],
                render_width=int(depth_cfg_raw["render_width"]),
                render_height=int(depth_cfg_raw["render_height"]),
                output_width=int(depth_cfg_raw["output_width"]),
                output_height=int(depth_cfg_raw["output_height"]),
                crop_region=tuple(int(x) for x in depth_cfg_raw["crop_region"]),
                depth_range=tuple(float(x) for x in depth_cfg_raw["depth_range"]),
                gaussian_blur=bool(depth_cfg_raw["gaussian_blur"]),
                flip_horizontal=bool(depth_cfg_raw["flip_horizontal"]),
                history_len=int(policy_cfg["depth_history"]),
                history_skip_frames=int(policy_cfg["depth_history_skip_frames"]),
            ),
        )

        self.action_clip = float(policy_cfg["action_clip"])
        self.obs_clip = float(policy_cfg["observation_clip"])
        self.warmup_policy_steps = int(policy_cfg["warmup_policy_steps"])
        self.log_dir = cfg.resolve("paths.log_dir")
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def reset(self) -> None:
        self.robot.write_default_pose(self.data, root_height=float(self.cfg.data["robot"]["root_height"]))
        self.last_action[:] = 0.0
        self.target_pos[:] = self.robot.default_pos
        self.obs_builder.reset(self.last_action)
        self.depth.reset()
        self.mujoco.mj_forward(self.model, self.data)

    def step_low_level(self) -> None:
        qpos = self.robot.read_qpos(self.data)
        qvel = self.robot.read_qvel(self.data)
        tau = pd_torque(self.target_pos, qpos, qvel, self.robot.kp, self.robot.kd, self.robot.effort_limit)
        self.robot.apply_torque(self.data, tau)
        self.mujoco.mj_step(self.model, self.data)

    def step_policy(self, policy_step: int) -> None:
        self.depth.capture_processed_frame(self.data)
        self.obs_builder.append(
            self.data,
            self.robot,
            self.command,
            self.last_action,
            imu_yaw_correction_rad=self.imu_yaw_correction_rad,
        )
        if policy_step < self.warmup_policy_steps:
            self.last_action[:] = 0.0
        else:
            proprio = self.obs_builder.build(clip=self.obs_clip)
            depth_obs = self.depth.depth_observation()
            self.last_action[:] = self.policy.infer(proprio, depth_obs, action_clip=self.action_clip)
        self.target_pos[:] = self.robot.action_to_target(self.last_action)

    def run(self) -> None:
        sim_cfg = self.cfg.data["simulation"]
        self.reset()
        self._write_mapping_table()
        log_path = self.log_dir / f"sim2sim_{int(time.time())}.csv"

        with log_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["t", "root_x", "root_y", "root_z", "action_min", "action_max", "action_mean"])
            if bool(sim_cfg["headless"]):
                self._run_headless(writer)
            else:
                self._run_viewer(writer)

        print(f"Wrote log: {log_path}")

    def _run_headless(self, writer: csv.writer) -> None:
        total_steps = int(self.duration / self.model.opt.timestep)
        renderer = None
        video_writer = None
        render_interval = max(1, int(round(1.0 / (self.video_fps * self.model.opt.timestep))))
        if self.record_video:
            if cv2 is None:
                raise RuntimeError("OpenCV is required for record_video=true")
            self.video_path.parent.mkdir(parents=True, exist_ok=True)
            renderer = self.mujoco.Renderer(self.model, width=self.video_width, height=self.video_height)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(str(self.video_path), fourcc, self.video_fps, (self.video_width, self.video_height))
            if not video_writer.isOpened():
                raise RuntimeError(f"Failed to open video writer: {self.video_path}")
            render_camera = self._make_render_camera()
        else:
            render_camera = -1

        for step in range(total_steps):
            self.step_low_level()
            if step % self.decimation == 0:
                policy_step = step // self.decimation
                self.step_policy(policy_step)
                self._log_row(writer)
            if video_writer is not None and renderer is not None and step % render_interval == 0:
                self._update_render_camera(render_camera)
                renderer.update_scene(self.data, camera=render_camera)
                rgb = renderer.render()
                video_writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        if video_writer is not None:
            video_writer.release()
            print(f"Wrote video: {self.video_path}")

    def _run_viewer(self, writer: csv.writer) -> None:
        import mujoco.viewer

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            step = 0
            start = time.time()
            while viewer.is_running() and time.time() - start < self.duration:
                tick = time.time()
                self.step_low_level()
                if step % self.decimation == 0:
                    policy_step = step // self.decimation
                    self.step_policy(policy_step)
                    self._log_row(writer)
                    if self.show_depth:
                        self._show_depth_window()
                viewer.sync()
                step += 1
                sleep_s = self.model.opt.timestep - (time.time() - tick)
                if sleep_s > 0:
                    time.sleep(sleep_s)

    def _log_row(self, writer: csv.writer) -> None:
        writer.writerow(
            [
                f"{self.data.time:.4f}",
                f"{self.data.qpos[0]:.6f}",
                f"{self.data.qpos[1]:.6f}",
                f"{self.data.qpos[2]:.6f}",
                f"{self.last_action.min():.6f}",
                f"{self.last_action.max():.6f}",
                f"{self.last_action.mean():.6f}",
            ]
        )

    def _write_mapping_table(self) -> None:
        path = self.log_dir / "joint_mapping.csv"
        path.write_text(self.robot.mapping_table() + "\n", encoding="utf-8")
        print(f"Wrote joint mapping: {path}")

    def _make_render_camera(self) -> Any:
        cam_cfg = self.video_camera_cfg
        if cam_cfg.get("mode", "free") != "tracking":
            return -1
        body_name = cam_cfg.get("body", "torso_link")
        body_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"Cannot track missing body: {body_name}")
        cam = self.mujoco.MjvCamera()
        self.mujoco.mjv_defaultCamera(cam)
        cam.type = self.mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = body_id
        cam.distance = float(cam_cfg.get("distance", 3.0))
        cam.azimuth = float(cam_cfg.get("azimuth", 135.0))
        cam.elevation = float(cam_cfg.get("elevation", -18.0))
        return cam

    def _update_render_camera(self, camera: Any) -> None:
        if not hasattr(camera, "lookat"):
            return
        offset = np.array(self.video_camera_cfg.get("lookat_offset", [0.0, 0.0, 0.0]), dtype=np.float32)
        camera.lookat[:] = self.data.qpos[:3] + offset

    def _show_depth_window(self) -> None:
        if cv2 is None:
            raise RuntimeError("OpenCV is required for show_depth=true")
        if self.depth.last_frame is None:
            return
        img = np.clip(self.depth.last_frame, 0.0, 1.0)
        img = (img * 255.0).astype(np.uint8)
        img = cv2.resize(img, (self.depth.cfg.output_width * 12, self.depth.cfg.output_height * 12), interpolation=cv2.INTER_NEAREST)
        img = cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS)
        cv2.imshow("policy depth 18x32 normalized", img)
        cv2.waitKey(1)
