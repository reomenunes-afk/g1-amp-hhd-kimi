"""MuJoCo joint/address adapter for the G1 policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .constants import (
    ACTION_SCALE_BY_NAME,
    EFFORT_LIMIT_BY_NAME,
    KD_BY_NAME,
    KP_BY_NAME,
    TRAINING_JOINT_ORDER,
    values_in_training_order,
)


@dataclass
class G1RobotAdapter:
    model: Any
    joint_names: list[str]
    qpos_addr: np.ndarray
    qvel_addr: np.ndarray
    actuator_addr: np.ndarray
    default_pos: np.ndarray
    action_scale: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    effort_limit: np.ndarray

    @classmethod
    def from_mujoco(cls, mujoco: Any, model: Any, default_joint_pos: dict[str, float]) -> "G1RobotAdapter":
        joint_names = list(TRAINING_JOINT_ORDER)
        qpos_addr = []
        qvel_addr = []
        actuator_addr = []

        for name in joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"Missing joint in MuJoCo model: {name}")
            qpos_addr.append(model.jnt_qposadr[jid])
            qvel_addr.append(model.jnt_dofadr[jid])

            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid < 0:
                raise ValueError(f"Missing actuator named like joint: {name}")
            actuator_addr.append(aid)

        missing = [name for name in joint_names if name not in default_joint_pos]
        if missing:
            raise ValueError(f"default_joint_pos missing entries: {missing}")

        return cls(
            model=model,
            joint_names=joint_names,
            qpos_addr=np.array(qpos_addr, dtype=np.int32),
            qvel_addr=np.array(qvel_addr, dtype=np.int32),
            actuator_addr=np.array(actuator_addr, dtype=np.int32),
            default_pos=np.array([default_joint_pos[name] for name in joint_names], dtype=np.float32),
            action_scale=np.array([ACTION_SCALE_BY_NAME[name] for name in joint_names], dtype=np.float32),
            kp=np.array(values_in_training_order(KP_BY_NAME), dtype=np.float32),
            kd=np.array(values_in_training_order(KD_BY_NAME), dtype=np.float32),
            effort_limit=np.array(values_in_training_order(EFFORT_LIMIT_BY_NAME), dtype=np.float32),
        )

    def read_qpos(self, data: Any) -> np.ndarray:
        return np.asarray(data.qpos[self.qpos_addr], dtype=np.float32)

    def read_qvel(self, data: Any) -> np.ndarray:
        return np.asarray(data.qvel[self.qvel_addr], dtype=np.float32)

    def write_default_pose(self, data: Any, root_height: float) -> None:
        data.qpos[:] = 0.0
        data.qvel[:] = 0.0
        data.qpos[2] = root_height
        data.qpos[3] = 1.0
        data.qpos[self.qpos_addr] = self.default_pos

    def action_to_target(self, action: np.ndarray) -> np.ndarray:
        return self.default_pos + action.astype(np.float32) * self.action_scale

    def apply_torque(self, data: Any, tau_training_order: np.ndarray) -> None:
        data.ctrl[self.actuator_addr] = tau_training_order

    def mapping_table(self) -> str:
        lines = ["idx,training_joint,qpos_addr,qvel_addr,actuator_addr,kp,kd,scale"]
        for i, name in enumerate(self.joint_names):
            lines.append(
                f"{i},{name},{self.qpos_addr[i]},{self.qvel_addr[i]},"
                f"{self.actuator_addr[i]},{self.kp[i]:.6g},{self.kd[i]:.6g},{self.action_scale[i]:.6g}"
            )
        return "\n".join(lines)
