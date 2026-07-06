"""Depth rendering and preprocessing matching the InstinctLab export."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - OpenCV is a runtime dependency.
    cv2 = None

from .history import HistoryBuffer


@dataclass
class DepthConfig:
    camera_name: str
    render_width: int
    render_height: int
    output_width: int
    output_height: int
    crop_region: tuple[int, int, int, int]
    depth_range: tuple[float, float]
    gaussian_blur: bool
    flip_horizontal: bool
    history_len: int
    history_skip_frames: int


class ParkourDepthProcessor:
    def __init__(self, mujoco: Any, model: Any, cfg: DepthConfig):
        self.mujoco = mujoco
        self.cfg = cfg
        self.camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cfg.camera_name)
        if self.camera_id < 0:
            raise ValueError(f"Missing depth camera: {cfg.camera_name}")
        self.renderer = mujoco.Renderer(model, width=cfg.render_width, height=cfg.render_height)
        self.renderer.enable_depth_rendering()
        self.raw_history = HistoryBuffer(cfg.history_len * cfg.history_skip_frames + 2, cfg.output_width * cfg.output_height)
        self.last_frame: np.ndarray | None = None

    def reset(self) -> None:
        self.raw_history.reset(np.ones(self.cfg.output_width * self.cfg.output_height, dtype=np.float32))
        self.last_frame = np.ones((self.cfg.output_height, self.cfg.output_width), dtype=np.float32)

    def capture_processed_frame(self, data: Any) -> np.ndarray:
        self.renderer.update_scene(data, camera=self.camera_id)
        raw = np.asarray(self.renderer.render(), dtype=np.float32)
        frame = self._preprocess_raw(raw)
        self.last_frame = frame.copy()
        self.raw_history.append(frame.reshape(-1))
        return frame

    def depth_observation(self) -> np.ndarray:
        history = self.raw_history.as_array()
        # Match delayed_visualizable_image: oldest selected frame first, latest last.
        offsets = list(range((self.cfg.history_len - 1) * self.cfg.history_skip_frames, -1, -self.cfg.history_skip_frames))
        selected = [history[-1 - off] for off in offsets]
        return np.stack(selected, axis=0).reshape(1, self.cfg.history_len, self.cfg.output_height, self.cfg.output_width)

    def _preprocess_raw(self, raw: np.ndarray) -> np.ndarray:
        top, bottom, left, right = self.cfg.crop_region
        cropped = raw[top : raw.shape[0] - bottom, left : raw.shape[1] - right]
        if cropped.shape != (self.cfg.output_height, self.cfg.output_width):
            if cv2 is None:
                raise RuntimeError("OpenCV is required when depth crop shape does not match output shape")
            cropped = cv2.resize(
                cropped,
                (self.cfg.output_width, self.cfg.output_height),
                interpolation=cv2.INTER_AREA,
            )
        if self.cfg.gaussian_blur:
            if cv2 is None:
                raise RuntimeError("OpenCV is required for gaussian_blur=true")
            cropped = cv2.GaussianBlur(cropped, (3, 3), sigmaX=1)
        if self.cfg.flip_horizontal:
            cropped = np.flip(cropped, axis=1)
        lo, hi = self.cfg.depth_range
        cropped = np.clip(cropped, lo, hi)
        cropped = (cropped - lo) / (hi - lo)
        return cropped.astype(np.float32)
