"""InstinctLab parkour policy observation builder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .history import HistoryBuffer
from .math_utils import apply_yaw_correction, projected_gravity_from_quat
from .robot import G1RobotAdapter


@dataclass
class ObservationScales:
    ang_vel: float = 0.25
    dof_pos: float = 1.0
    dof_vel: float = 0.05


class ParkourObservationBuilder:
    """Build the 768-d proprio input expected by actor.onnx."""

    def __init__(self, history_len: int, scales: ObservationScales | None = None):
        self.history_len = int(history_len)
        self.scales = scales or ObservationScales()
        self.base_ang_vel = HistoryBuffer(self.history_len, 3)
        self.projected_gravity = HistoryBuffer(self.history_len, 3)
        self.velocity_commands = HistoryBuffer(self.history_len, 3)
        self.joint_pos = HistoryBuffer(self.history_len, 29)
        self.joint_vel = HistoryBuffer(self.history_len, 29)
        self.actions = HistoryBuffer(self.history_len, 29)
        self.reset(np.zeros(29, dtype=np.float32))

    def reset(self, initial_action: np.ndarray) -> None:
        self.base_ang_vel.reset()
        self.projected_gravity.reset(np.array([0.0, 0.0, -1.0], dtype=np.float32))
        self.velocity_commands.reset()
        self.joint_pos.reset()
        self.joint_vel.reset()
        self.actions.reset(initial_action)

    def append(
        self,
        data: Any,
        robot: G1RobotAdapter,
        command: np.ndarray,
        last_action: np.ndarray,
        imu_yaw_correction_rad: float = 0.0,
    ) -> None:
        quat = np.asarray(data.qpos[3:7], dtype=np.float32)
        quat = apply_yaw_correction(quat, imu_yaw_correction_rad)
        ang_vel = np.asarray(data.qvel[3:6], dtype=np.float32) * self.scales.ang_vel
        qpos = (robot.read_qpos(data) - robot.default_pos) * self.scales.dof_pos
        qvel = robot.read_qvel(data) * self.scales.dof_vel

        self.base_ang_vel.append(ang_vel)
        self.projected_gravity.append(projected_gravity_from_quat(quat))
        self.velocity_commands.append(np.asarray(command, dtype=np.float32))
        self.joint_pos.append(qpos)
        self.joint_vel.append(qvel)
        self.actions.append(np.asarray(last_action, dtype=np.float32))

    def build(self, clip: float = 100.0) -> np.ndarray:
        obs = np.concatenate(
            [
                self.base_ang_vel.flatten(),
                self.projected_gravity.flatten(),
                self.velocity_commands.flatten(),
                self.joint_pos.flatten(),
                self.joint_vel.flatten(),
                self.actions.flatten(),
            ],
            axis=0,
        )
        if obs.shape != (768,):
            raise RuntimeError(f"Unexpected proprio observation shape: {obs.shape}")
        return np.clip(obs, -clip, clip).astype(np.float32)
