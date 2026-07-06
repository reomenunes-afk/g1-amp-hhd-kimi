"""Two-stage ONNX inference for the exported parkour policy."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class ParkourOnnxPolicy:
    def __init__(self, actor_path: str | Path, depth_encoder_path: str | Path):
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - runtime dependency.
            raise RuntimeError("onnxruntime is required to run exported ONNX policies") from exc

        providers = ["CPUExecutionProvider"]
        self.encoder = ort.InferenceSession(str(depth_encoder_path), providers=providers)
        self.actor = ort.InferenceSession(str(actor_path), providers=providers)
        self.encoder_input = self.encoder.get_inputs()[0].name
        self.actor_input = self.actor.get_inputs()[0].name

        self.encoder_out_dim = self._infer_encoder_out_dim()
        actor_shape = self.actor.get_inputs()[0].shape
        if isinstance(actor_shape[-1], int) and actor_shape[-1] != 768 + self.encoder_out_dim:
            raise RuntimeError(
                f"actor input dim mismatch: actor expects {actor_shape[-1]}, "
                f"proprio+depth gives {768 + self.encoder_out_dim}"
            )

    def _infer_encoder_out_dim(self) -> int:
        shape = self.encoder.get_inputs()[0].shape
        dims = [1 if not isinstance(dim, int) else dim for dim in shape]
        dummy = np.zeros(dims, dtype=np.float32)
        out = self.encoder.run(None, {self.encoder_input: dummy})[0]
        return int(out.reshape(out.shape[0], -1).shape[1])

    def infer(self, proprio_obs: np.ndarray, depth_obs: np.ndarray, action_clip: float) -> np.ndarray:
        proprio_obs = proprio_obs.reshape(1, -1).astype(np.float32)
        depth_obs = depth_obs.astype(np.float32)
        depth_latent = self.encoder.run(None, {self.encoder_input: depth_obs})[0]
        depth_latent = depth_latent.reshape(depth_latent.shape[0], -1).astype(np.float32)
        actor_input = np.concatenate([proprio_obs, depth_latent], axis=1).astype(np.float32)
        action = self.actor.run(None, {self.actor_input: actor_input})[0].reshape(-1)
        return np.clip(action, -action_clip, action_clip).astype(np.float32)
