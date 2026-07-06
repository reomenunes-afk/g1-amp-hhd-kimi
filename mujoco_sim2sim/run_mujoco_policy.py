#!/usr/bin/env python3
"""Run the exported InstinctLab G1 parkour ONNX policy in MuJoCo."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim2sim.runner import MujocoSim2SimRunner, RunnerConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "g1_parkour_version_policy.yaml"),
        help="Path to sim2sim YAML config.",
    )
    parser.add_argument("--headless", action="store_true", help="Override config and run without viewer.")
    parser.add_argument("--show-depth", action="store_true", help="Show the processed policy depth image.")
    parser.add_argument("--record-video", action="store_true", help="Render an MP4 while running.")
    parser.add_argument("--duration", type=float, default=None, help="Override simulation duration in seconds.")
    args = parser.parse_args()

    cfg = RunnerConfig.load(args.config)
    if args.headless:
        cfg.data["simulation"]["headless"] = True
    if args.show_depth:
        cfg.data["simulation"]["show_depth"] = True
    if args.record_video:
        cfg.data["simulation"]["record_video"] = True
        cfg.data["simulation"]["headless"] = True
    if args.duration is not None:
        cfg.data["simulation"]["duration"] = args.duration

    runner = MujocoSim2SimRunner(cfg)
    runner.run()


if __name__ == "__main__":
    main()
