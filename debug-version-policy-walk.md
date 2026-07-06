# Debug Session: version-policy-walk

Status: [OPEN]

## Symptom
The MuJoCo G1 robot falls when running the depth version policy mounted on `WalkNet1Action`.

## Goal
Make the simulation run fully automatically and use runtime evidence to debug until the policy can walk normally or the remaining blocker is proven.

## Hypotheses
1. The policy is enabled before `ParkourDepthImage` has valid camera history, so the actor sees zero/near depth and behaves as if stepping onto an obstacle.
2. The reset/autostart timing does not match the stable manual flow and puts the robot outside the training distribution before policy takeover.
3. A joint/action order mismatch remains between CtrlZ device order and training order.
4. Proprioception or depth history order/fill behavior differs from the training exporter.
5. Simulation control parameters, clipping, or dynamics differ enough from training to destabilize an otherwise valid policy.

## Evidence Log
- Prior run with `ForceDepthValue=-1` showed clean IMU at takeover but `depth_min=0 depth_max=0 depth_mean=0`, supporting hypothesis 1.
- Code inspection on 2026-07-02: actor `IoBinding__` is created in `AbstractNetInferenceWorker::TaskCreate()`, after the derived `UnitreeRlVersionInferenceWorker` constructor has already pushed `ActorInputTensor` and `OutputTensor` into `InputOrtTensors__` / `OutputOrtTensors__`. This rejects the current variant of hypothesis "actor session is bound before version tensors exist".

## Next Step
Run the fully automated simulation and compare runtime evidence around takeover, input statistics, raw action statistics, IMU state, and final pose. If the actor input path behaves normally, continue with hypotheses around action semantics, default pose / asset mismatch, and policy takeover state.

## 2026-07-03 Evidence Update
- Baseline with fixed depth (`ForceDepthValue=1`) still falls quickly, rejecting the hypothesis that visual depth content is the primary cause.
- `ImuYawCorrectionRad=+1.5708` makes early actor output much worse (`raw_action_max` around 11 by tick 51), rejecting that yaw direction.
- `ImuYawCorrectionRad=-1.5708` significantly reduces angular-velocity divergence later in the run, but the robot still settles in a fallen/tilted posture (`projected_gravity` stays far from `[0,0,-1]`).
- Training parkour `velocity_commands` observation has no scale in `parkour_env_cfg.py`; deployment was multiplying commands by `[0,0,0.25]`, so actor did not see the intended forward command. Patch version worker to use unscaled command history while keeping base angular velocity scale at `0.25`.

## 2026-07-02 Autonomous Debug Continuation
User requested a fully automated debug loop without frontend keyboard operation.

Additional falsifiable hypotheses for the next loop:
1. The current MJCF root/body hierarchy differs from the training `g1_29dof_torsobase_popsicle` asset enough that the same default pose is not dynamically equivalent.
2. The robot is already in an unstable contact/base state before actor takeover, so the version policy starts outside its training distribution.
3. The actor output is semantically plausible in training order, but postprocess/default/action-scale produces invalid target poses in CtrlZ device order.
4. The depth encoder output is numerically valid, but depth observation content or frame orientation biases the policy toward stair/high-step behavior.
5. A minimal training-asset scene with equivalent sensors and actuators will improve pre-policy standing and reduce early actor divergence if the asset mismatch is dominant.

Constraint: first existing-code change in this continuation must be instrumentation only. Business logic changes follow only after runtime evidence narrows the root cause.

## 2026-07-03 Autonomous Loop Notes
- Rechecked InstinctLab `PolicyCfg`: policy proprio does **not** include `base_lin_vel`; `play.py` lists it in `proprio_slice`, but `get_subobs_size()` only counts components present in `obs_segments`, so current 768-d proprio + 128-d depth latent is consistent with `actor.onnx`.
- Rechecked `beyondmimic_action_scale` ordering. The JSON `action_scale` is in CtrlZ `MotorVec` / `JOINT_ID_MAP` order, not raw URDF order. Therefore `TrainingActionScale = DeviceToTrain(ActionScale)` is correct; the temporary direct-use change was reverted.
- A/B: 3-second reset/default-pose wait falls before policy takeover. This is not a valid policy test because fixed joint targets alone do not balance the floating-base biped.
- A/B after scale-order verification:
  - fixed depth (`ForceDepthValue=1.0`) still falls and can produce more aggressive first actions, so depth content alone is not the primary root cause.
  - zero command (`WalkCmd=[0,0,0]`) still falls, so the issue is not just forward-command direction.
  - `ImuYawCorrectionRad=-1.5708` with corrected scale did not improve stability enough; frame yaw rotation alone is not sufficient.
- Joint axes in MJCF and training URDF match for all 29 revolute joints; no obvious joint-axis sign inversion was found.
- Current remaining high-value suspects:
  1. IsaacLab joint/action order may still differ from raw URDF if `find_joints(".*")` ordering is not raw asset order.
  2. IMU/body-frame convention may need a fuller basis transform than yaw-only correction.
  3. MuJoCo actuator/contact dynamics differ from Isaac delayed actuators enough to amplify early policy errors.
  4. Need model-shape and per-segment actor-input instrumentation to compare exact exported input assumptions.
- Operational note: repeated `xvfb-run` timeouts left several unkillable/stale Xvfb processes in this container session. Avoid relying on fresh `xvfb-run` until display resources are recovered; use code/static checks or a clean shell/session if available.

## 2026-07-03 Continued Automated Tests
- Recovered automated `xvfb-run` execution; `timeout` exit code `124` remains expected for bounded runs.
- A/B `TrainToDeviceIndex` using the IsaacLab `asset.data.joint_names` order documented in `instinctlab/assets/unitree_g1.py` made the run worse: by tick 51 `projected_gravity` was approximately `[-0.998, 0.022, 0.065]`. Reverted to the deploy `joint_ids_map` / CtrlZ `JOINT_ID_MAP` mapping.
- Confirmed Unitree deploy examples export `joint_ids_map: [0, 6, 12, 1, 7, 13, ...]`, matching current CtrlZ `JOINT_ID_MAP`. This lowers the likelihood of a remaining action/joint order issue.
- Added a `UseAlterImu` test switch to feed `secondary_imu`/torso IMU rpy+gyro into the version policy. It changed the failure shape but did not solve falling; reverted runtime config to `UseAlterImu=false`.
- A/B root height from pelvis `z=0.82` to `z=0.846` based on torso-base root `z=0.9` did not solve falling; reverted to `0.82`.
- Found that local `CheckPoints/version_policy/*.onnx` do not match `/home/bytedance/Documents/Instinclab-stable/Dataset/checkpoints/parkour_onboard_preview_stair/exported/*.onnx`:
  - local actor sha256 `57890ffe...`, size about 4.38 MB.
  - parkour exported actor sha256 `81b1af17...`, size about 2.51 MB.
  - local depth encoder also differs from exported depth encoder.
- Running the original `parkour_onboard_preview_stair/exported` ONNX was worse than the local version, with early raw actions around `[-6.45, 4.34]` by tick 51.
- Running `stand_onboard/exported` with zero command also fell quickly. Since `stand_onboard` and `parkour_onboard_preview_stair` have identical `env.yaml`, this points away from a parkour-only weight issue and toward deployment/physics/sensor mismatch.
- Rechecked training default joint pose in `env.yaml`; current default pose is close to training values (`hip_pitch=-0.312`, `knee=0.669`, `ankle_pitch=-0.363`, `elbow=0.6`), so a gross default-pose mismatch is unlikely.

Current likely blockers:
1. Missing exact `deploy.yaml`/training config for local `CheckPoints/version_policy` ONNX.
2. MuJoCo dynamics/contact/armature/friction mismatch versus IsaacLab training asset.
3.

## 2026-07-03 Sideways-Walk Config Sync
- User reported the robot can now barely walk but moves sideways. User also pointed out parameters moved into `settings/bitbot_mujoco.xml` and provided play-stair videos under `/home/bytedance/Downloads/videos/play_stair`.
- Static/video inspection only; did **not** run sim per user constraint.
- Confirmed `/home/bytedance/Downloads/videos/play_stair/model_40000-step-0.mp4` shows the policy walking forward toward stairs in the training/play setup, so the learned policy direction is forward, not sideways.
- Confirmed `CheckPoints/version_policy/*.onnx` are byte-identical to `/home/bytedance/Documents/20260629_172947/exported/*.onnx`; matching training config is `/home/bytedance/Documents/20260629_172947/params/env.yaml`.
- Training command is `base_velocity` with `only_positive_lin_vel_x: true`; play config fixes `lin_vel_x=FORWARD_SPEED` and `lin_vel_y=0`, so deployment must keep forward command in observation index 0.
- Found active `settings/CtrlConfig.json` was not synchronized with `settings/CtrlConfig_mujoco.json` / training defaults:
  - `Workers.MotorControl.DefaultPosition` was not the training default pose used by the version policy default-joint offset.
  - `WalkNet1.Preprocess.ForceDepthValue` was `1.0`, forcing constant depth instead of real camera depth.
  - `FlipDepthHorizontal` was absent from the active config even though `user_func.cpp` implements it.
  - `WalkCmd.DefaultValue` was `[0, 0, 0]`, not forward walking `[0.6, 0, 0]`.
- Applied active config sync in `settings/CtrlConfig.json`:
  - training `MotorControl.DefaultPosition`
  - `ForceDepthValue=-1.0`
  - `FlipDepthHorizontal=true`
  - `InvertAngVelZ=false`
  - `WalkCmd.DefaultValue=[0.6,0,0]`
  - `AutoDebug.DepthWarmupFrames=0` to match `CtrlConfig_mujoco.json`
- Build-only verification passed with `cmake --build build -j2`; no simulation was launched.

Next test request for user: run the usual MuJoCo flow once with active `settings/CtrlConfig.json`. If it still walks sideways, keep `WalkCmd=[0.6,0,0]` and report whether the robot body is yawed 90 degrees while the world path is straight, or whether the body faces forward but lateral velocity is dominant. That distinction determines whether the next fix is yaw-frame transform or command/body-frame mapping.

