# Debug Session: policy-cpp-alignment

Status: [OPEN]

## Problem

The same Unitree G1 parkour policy works in the Python MuJoCo sim2sim reference but fails to walk correctly in the C++/bitbot deployment under `/home/bytedance/Documents/unitree_bitbot_hhd`.

## Reference

- Python reference: `/home/bytedance/Documents/unitree_bitbot_hhd/mujoco_sim2sim`
- C++ project: `/home/bytedance/Documents/unitree_bitbot_hhd`
- Policy: actor + depth encoder ONNX models

## Hypotheses

1. Depth encoder input tensor shape/layout is correct at the wrapper level, but the C++ depth history content or ordering differs from Python.
2. C++ starts the policy before the raycaster depth image/history is warmed up, so early invalid depth corrupts the rollout.
3. C++ proprioception still differs from Python in root angular velocity or projected gravity coordinates.
4. C++ postprocess/control differs from Python in policy warmup, action hold, or PD target behavior.
5. C++ actor input concatenation produces a different depth latent/action despite similar high-level depth statistics.

## Evidence Log

- `WarpOrtTensor` passes static shape pointers directly to ONNXRuntime; `Tensor<1,8,18,32>` should be bound as `(1,8,18,32)`.
- Need runtime evidence for actual tensor values, latent values, and action values.
- ONNXRuntime reports actor input `(1,896)`, actor output `(1,29)`, depth encoder input `(1,8,18,32)`, depth encoder output `(1,128)`.
- Short C++ run with current config switched to policy at `camera_frames=0/0`; first `VersionPolicyDebug` reported `depth_min=0 depth_max=0 depth_mean=0`.
- Camera worker later reported `camera_frames=37 history_size=37/37 ready=1`, so the AutoDebug switch occurs before the first valid depth history.

## Fix Attempt 1

Wait for the same depth history length used by the camera worker before enabling `InferWalkTask`.

Result: rejected. The policy switch waited until `camera_frames=37`, but the robot had already pitched at the default target; then `PolicyWarmupSteps=40` added another warmup period. The rollout fell much earlier than the previous run. This proves pre-policy depth wait is not the correct alignment; depth should warm while the policy task is already running, as in Python.

## Confirmed Causes

1. C++ version policy postprocess used stale/misordered `action_scale` and joint clip arrays. The JSON arrays were neither the training joint order nor the bitbot device order, so `DeviceToTrain()` produced wrong action amplitudes.
2. `ActionManagementWorker` overwrote MuJoCo joint KP/KD with stale/misordered arrays during policy switch. The `bitbot_mujoco.xml` device KP/KD were correct, but the ActionManager replacement was wrong.
3. After fixing scale/limits and KP/KD order, the rollout climbs the 20 cm / 30 cm stair sequence. Remaining y drift is reduced by internal heading hold.

## Fixes Kept

- `UnitreeRlVersionInferenceWorker` now uses training-order reference constants for action scale and joint limits for this exported version policy.
- MuJoCo `ActionManager.MotorProperties` are overridden to device-order KP/KD before worker creation.
- `CtrlConfig_mujoco.json` sets `HeadingHoldKp=2.0` for internal straight-line yaw correction.

## Verification

- Build passed with `cmake --build build -j$(nproc)`.
- With fixed action scale and KP/KD, 12 s run reached about `x=5.0, z=1.86` before heading hold tuning.
- With `HeadingHoldKp=2.0`, 10 s run reached `x=3.86, y=0.42, z=1.99` in the visible log window and stayed on the stair/top-platform trajectory.
