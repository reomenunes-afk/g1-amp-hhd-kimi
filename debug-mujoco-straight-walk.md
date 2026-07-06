# Debug Session: mujoco-straight-walk

Status: [OPEN]

## 2026-07-03 MuJoCo Runtime Evidence

### Runs
- Original `WalkCmd=[0.6,0,0]`, original root yaw: 18s run moved mostly sideways.
  - delta = `(-0.187, -7.346, 0.003)`
  - heading = `-91.46 deg`
  - `|dy/dx| = 39.228`
- Root yaw `+60 deg`, no heading hold: short run improved, but 30s run drifted later.
  - 30s delta = `(7.439, -9.598, -0.018)`
  - heading = `-52.22 deg`
  - drift starts after about `policy_steps=4250`.
- Root yaw `+60 deg` plus heading hold (`HeadingHoldKp=0.03`, `HeadingHoldMaxYawRate=0.05`): 30s run stayed forward.
  - delta = `(12.348, -1.558, 0.004)`
  - heading = `-7.19 deg`
  - `|dy/dx| = 0.126`
  - height stayed around `0.77~0.80 m` at the end.

### Rejected A/B
- `InvertAngVelZ=true`: robot fell, base height around `0.06 m`.
- `UseAlterImu=true`: robot fell/unstable, base height around `0.1 m` later.
- `ForceDepthValue=1.0`: robot fell and stopped, so real depth remains necessary.
- `FlipDepthHorizontal=false`: trajectory worsened into `-X/-Y`; keep `true`.
- Constant yaw command `+0.05`: eventually fell; dynamic heading hold is safer.

### Applied Fix
- `models/g1_29dof.xml`: set root torso yaw to `+60 deg` using `quat="0.86602540 0 0 0.50000000"`.
- `UnitreeRlVersionInferenceWoker.hpp`: added configurable heading hold on the command z observation before actor input.
- `settings/CtrlConfig_mujoco.json` and `settings/CtrlConfig.json`: set `HeadingHoldKp=0.03`, `HeadingHoldMaxYawRate=0.05`, keep `WalkCmd=[0.6,0,0]`, `ForceDepthValue=-1`, `FlipDepthHorizontal=true`, `UseAlterImu=false`, `InvertAngVelZ=false`.

Status: current MuJoCo runtime result is forward walking with small residual heading error (~7 deg over 30s), substantially fixed versus original sideways walking.

## 2026-07-03 PoseVelocityCommand Alignment Update

User correctly pointed out this should be a configuration/command-semantics issue because the policy walks straight in InstinctLab sim.

### Key finding
InstinctLab does not feed a constant `[vx, vy, 0]` command for this parkour policy. The training command generator is `PoseVelocityCommand`:
- `target_direction = atan2(target_y - root_y, target_x - root_x)`
- `heading_command_w = wrap_to_pi(target_direction - robot_heading_w)`
- `vel_command_b[:, 2] = heading_command_w * heading_control_stiffness`
- training config has `heading_control_stiffness=2.0` and `ang_vel_z=(-1.0, 1.0)`.

In MuJoCo deployment, a fixed `WalkCmd=[0.6,0,0]` removed this heading command semantics, so the policy slowly yawed away and then walked sideways. The correct deployment approximation is to hold the heading at policy-entry direction for the simplified MuJoCo course.

### Final runtime A/B
- Original sideways baseline: `delta=(-0.187,-7.346,0.003)`, heading `-91.46 deg`, `|dy/dx|=39.228`.
- Root yaw `+60 deg`, no heading command: 30s drifted, heading `-52.22 deg`.
- Initial-heading hold `Kp=0.1`, `MaxYawRate=0.2`: 45s `delta=(19.566,-7.285,0.016)`, heading `-20.42 deg`; still too weak for long run.
- Initial-heading hold `Kp=0.2`, `MaxYawRate=0.4`: 45s `delta=(19.979,-2.308,0.026)`, heading `-6.59 deg`.
- Initial-heading hold `Kp=0.3`, `MaxYawRate=0.6`: 45s `delta=(20.076,-1.043,-0.030)`, heading `-2.98 deg`, `|dy/dx|=0.052`, height stayed stable around `0.75~0.81 m`.

### Current applied configuration
- `models/g1_29dof.xml`: root `torso_link` starts with `quat="0.86602540 0 0 0.50000000"` (`+60 deg` yaw), matching the MuJoCo course/robot initial direction found by runtime A/B.
- `settings/CtrlConfig_mujoco.json` and `settings/CtrlConfig.json`:
  - `HeadingHoldKp=0.3`
  - `HeadingHoldMaxYawRate=0.6`
  - no fixed `HeadingHoldTargetYawRad`, so the policy-entry heading is used as target.
  - keep `ForceDepthValue=-1.0`, `FlipDepthHorizontal=true`, `UseAlterImu=false`, `InvertAngVelZ=false`, `WalkCmd=[0.6,0,0]`.

Status: current MuJoCo runtime result is stable forward walking over 45s with only about 3 deg integrated heading error. This is no longer a visual-only fix; it was verified from MuJoCo base trajectory logs.

