#!/usr/bin/env python3
import argparse
from pathlib import Path

import importlib.util
import numpy as np


def load_replay_module():
    module_path = Path(__file__).with_name("replay_trajectory_mujoco.py")
    spec = importlib.util.spec_from_file_location("replay_trajectory_mujoco", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def describe_axis(name, vec):
    labels = ["MuJoCo X", "MuJoCo Y", "MuJoCo Z"]
    axis = int(np.argmax(np.abs(vec)))
    sign = "+" if vec[axis] >= 0 else "-"
    return f"{name} -> {sign}{labels[axis]}   vector={vec.tolist()}"


def main():
    parser = argparse.ArgumentParser(description="Print ORB-SLAM3 to MuJoCo coordinate basis mapping.")
    parser.add_argument("--basis", default="opencv_to_mujoco", help="identity, opencv_to_mujoco, ros_to_mujoco, or 9 row-major numbers.")
    args = parser.parse_args()

    replay = load_replay_module()
    basis = replay.parse_basis(args.basis)
    print(f"basis: {args.basis}")
    print("Matrix B maps ORB/world vectors into MuJoCo/world vectors:")
    print(basis)
    print()
    print(describe_axis("ORB +X", basis @ np.asarray([1.0, 0.0, 0.0])))
    print(describe_axis("ORB +Y", basis @ np.asarray([0.0, 1.0, 0.0])))
    print(describe_axis("ORB +Z", basis @ np.asarray([0.0, 0.0, 1.0])))
    print()
    print(f"det(B) = {np.linalg.det(basis):.6f}")
    if np.linalg.det(basis) < 0:
        print("warning: determinant is negative; this basis mirrors handedness.")


if __name__ == "__main__":
    main()
