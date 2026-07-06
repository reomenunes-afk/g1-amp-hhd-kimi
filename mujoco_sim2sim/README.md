# G1 Parkour MuJoCo Sim2Sim

This is a standalone MuJoCo runner for the InstinctLab G1 parkour policy exported to:

- `CheckPoints/version_policy/actor.onnx`
- `CheckPoints/version_policy/0-depth_encoder.onnx`

It is intentionally separate from the Bitbot/CtrlZ state machine so the policy can be tested with a minimal, auditable sim2sim loop.

## What It Matches

- Training run: `/home/bytedance/Documents/20260629_172947`
- Policy interval: `0.02s`
- Proprio observation: `8 * (base_ang_vel, projected_gravity, command, joint_pos, joint_vel, last_action) = 768`
- Depth observation: `8 x 18 x 32`
- Depth camera render source: `64 x 36`, crop `[top=18, bottom=0, left=16, right=16]`
- Command used by `play_stair`: `[0.6, 0.0, 0.0]`
- Joint access: explicit MuJoCo joint names mapped into InstinctLab training joint order

## Static Check

This does not start simulation:

```bash
cd /home/bytedance/Documents/unitree_bitbot_hhd/mujoco_sim2sim
python3 check_static.py
```

## Run Later On A Machine That Can Simulate

Viewer:

```bash
cd /home/bytedance/Documents/unitree_bitbot_hhd/mujoco_sim2sim
python3 run_mujoco_policy.py
```

Headless:

```bash
python3 run_mujoco_policy.py --headless --duration 20
```

The runner writes:

- `log/mujoco_sim2sim/joint_mapping.csv`
- `log/mujoco_sim2sim/sim2sim_<timestamp>.csv`

## Debugging Notes

If the robot moves sideways while facing forward, inspect command/body-frame mapping first. If it walks in a straight world direction but the body is yawed 90 degrees, inspect `robot.imu_yaw_correction_rad` and the MuJoCo camera/body basis.

The framework does not add training-time observation noise. For sim2sim debugging this is intentional: first reproduce deterministic observation semantics, then add noise only after baseline behavior is understood.
