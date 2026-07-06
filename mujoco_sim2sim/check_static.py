#!/usr/bin/env python3
"""Static checks that do not start MuJoCo simulation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim2sim.constants import ACTION_SCALE_BY_NAME, TRAINING_JOINT_ORDER
from sim2sim.runner import RunnerConfig


def main() -> None:
    cfg = RunnerConfig.load(ROOT / "configs" / "g1_parkour_version_policy.yaml")
    required = [
        cfg.resolve("paths.xml"),
        cfg.resolve("paths.raycaster_plugin"),
        cfg.resolve("paths.actor_onnx"),
        cfg.resolve("paths.depth_encoder_onnx"),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    defaults = cfg.data["robot"]["default_joint_pos"]
    missing_defaults = [name for name in TRAINING_JOINT_ORDER if name not in defaults]
    if missing_defaults:
        raise RuntimeError(f"Missing default joints: {missing_defaults}")

    missing_scales = [name for name in TRAINING_JOINT_ORDER if name not in ACTION_SCALE_BY_NAME]
    if missing_scales:
        raise RuntimeError(f"Missing action scales: {missing_scales}")

    policy = cfg.data["policy"]
    proprio_dim = int(policy["proprio_history"]) * (3 + 3 + 3 + 29 + 29 + 29)
    depth_dim = int(policy["depth_history"]) * int(cfg.data["depth"]["output_height"]) * int(cfg.data["depth"]["output_width"])
    print(f"static check ok: proprio_dim={proprio_dim}, depth_dim={depth_dim}")
    print(f"xml={cfg.resolve('paths.xml')}")
    print(f"actor={cfg.resolve('paths.actor_onnx')}")
    print(f"depth_encoder={cfg.resolve('paths.depth_encoder_onnx')}")


if __name__ == "__main__":
    main()
