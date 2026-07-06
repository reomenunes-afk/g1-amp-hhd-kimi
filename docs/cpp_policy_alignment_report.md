# C++ Policy 对齐问题排查报告

## 1. 背景

本报告记录 `/home/bytedance/Documents/unitree_bitbot_hhd` 中 C++/bitbot 部署代码与 Python MuJoCo sim2sim 参考实现之间的对齐排查过程。

同一个 Unitree G1 parkour policy 在 Python MuJoCo sim2sim 中表现较好，能够完成 20 cm 高、30 cm 深台阶场景；但在 C++/bitbot 部署代码中，最初表现为走路不稳定、侧向漂移明显，甚至无法形成正常 gait。

这说明问题大概率不在 policy 本身，而在 C++ 侧的部署链路：

- ONNX 模型输入输出是否一致
- proprioception 观测是否一致
- depth encoder 输入是否一致
- 关节顺序映射是否一致
- action scale / joint limit 是否一致
- PD 控制参数是否一致
- policy warmup / depth history 时序是否一致

最终确认的主要问题是：

1. C++ version policy 后处理使用了错误顺序的 `action_scale` 和 joint limit。
2. `ActionManagementWorker` 在 policy 切换时又用错误顺序的 KP/KD 覆盖了 MuJoCo joint 的正确 PD 参数。
3. 在修复以上问题后，剩余的直行偏航漂移通过内部 heading hold 明显改善。

## 2. 参考实现

### 2.1 Python 参考路径

Python MuJoCo sim2sim 参考实现位于：

```text
/home/bytedance/Documents/unitree_bitbot_hhd/mujoco_sim2sim
```

核心文件：

```text
mujoco_sim2sim/run_mujoco_policy.py
mujoco_sim2sim/sim2sim/runner.py
mujoco_sim2sim/sim2sim/robot.py
mujoco_sim2sim/sim2sim/observations.py
mujoco_sim2sim/sim2sim/depth.py
mujoco_sim2sim/sim2sim/constants.py
mujoco_sim2sim/configs/g1_parkour_version_policy.yaml
```

Python 参考的关键行为：

- MuJoCo timestep: `0.002`
- policy decimation: `10`
- policy dt: `0.02`
- command: `[0.6, 0.0, 0.0]`
- proprio history: `8`
- depth history: `8`
- depth history skip: `5`
- warmup policy steps: `40`
- depth input shape: `(1, 8, 18, 32)`
- actor input shape: `(1, 896)`

### 2.2 C++ 部署路径

C++/bitbot 项目位于：

```text
/home/bytedance/Documents/unitree_bitbot_hhd
```

核心文件：

```text
user_func.cpp
settings/CtrlConfig_mujoco.json
settings/bitbot_mujoco.xml
CtrlZ/CtrlZ/Workers/NN/UnitreeRlVersionInferenceWoker.hpp
CtrlZ/CtrlZ/Workers/NN/CommonLocoInferenceWorker.hpp
CtrlZ/CtrlZ/Workers/NN/AbstractInferenceWorker.hpp
CtrlZ/CtrlZ/Utils/TensorType.hpp
mujoco_camera/src/mujoco_camera.cpp
```

## 3. 初始现象

Python 参考实现中，policy 能够在 MuJoCo 里稳定前进并上台阶。之前验证 8 秒左右的运行结果大致为：

```text
root_x ≈ 4.63
root_y ≈ 0.16
root_z ≈ 2.03
```

C++ 初始表现：

- policy 能正常加载
- depth encoder 和 actor 都能运行
- 输出 action 数值看起来不是 NaN，也不是完全异常
- 机器人却不能稳定形成与 Python 一致的 gait
- 早期表现为侧漂、姿态不稳、上台阶失败

这类现象通常意味着：

- 模型没坏
- 推理没崩
- 但是输入/输出语义被解释错了

也就是 deployment mismatch，而不是 training/policy quality 问题。

## 4. 排查假设

排查时先列了几个可验证假设。

### 假设 1：ONNX 输入 shape 或内存 layout 错

如果 C++ 把 depth input 从 `(1, 8, 18, 32)` 传成其他 layout，或者 actor input 拼接顺序错，会出现数值统计正常但 latent/action 完全错。

### 假设 2：depth history 时序不一致

C++ 使用 MuJoCo raycaster sensor，Python 使用 `mujoco.Renderer.enable_depth_rendering()`。两者单帧处理虽然接近，但启动时 raycaster 可能需要 warmup。

如果 policy 一开始拿到全 0 或全 1 depth history，可能导致 early rollout 偏离。

### 假设 3：base angular velocity / projected gravity 坐标系不一致

Python 使用：

```python
data.qvel[3:6]
```

作为 base angular velocity，并使用 root quaternion 计算 projected gravity。

C++ 侧如果使用 IMU site gyro 或其他坐标系，可能造成观测不一致。

### 假设 4：关节顺序映射错误

训练顺序和 bitbot 设备顺序不一致。如果 qpos/qvel/action/default_pos/action_scale/joint_limit 任意一个数组顺序混用，policy 输出语义会错。

### 假设 5：PD 控制参数不一致

Python 中直接按训练顺序计算 torque：

```python
tau = kp * (target_pos - qpos) - kd * qvel
tau = clip(tau, effort_limit)
```

C++ 中 `MujocoJoint` 使用 position mode，由 joint device 自己做 PD。如果 KP/KD 或 torque limit 不一致，也会造成 sim2sim 行为偏差。

## 5. 排查过程

### 5.1 确认 ONNX 模型和输入输出 shape

首先检查 actor 和 depth encoder 的 ONNX shape。

确认结果：

```text
actor.onnx
  input:  ("input", [1, 896], tensor(float))
  output: ("output", [1, 29], tensor(float))

0-depth_encoder.onnx
  input:  ("input", [1, 8, 18, 32], tensor(float))
  output: ("output", [1, 128], tensor(float))
```

C++ 中对应 tensor：

```cpp
z::math::Tensor<InferencePrecision, 1, DEPTH_HISTORY_LENGTH, DEPTH_HEIGHT, DEPTH_WIDTH> DepthInputTensor;
z::math::Tensor<InferencePrecision, 1, ACTOR_INPUT_TENSOR_LENGTH> ActorInputTensor;
```

`WarpOrtTensor()` 的实现：

```cpp
return Ort::Value::CreateTensor<InferencePrecision>(
    this->MemoryInfo__,
    Tensor.data(),
    Tensor.size(),
    Tensor.shape_ptr(),
    Tensor.num_dims()
);
```

`TensorType.hpp` 中的索引计算是 row-major：

```cpp
index = i0 * stride0 + i1 * stride1 + ...
```

因此 C++ wrapper 层没有发现 shape 反转或内存 layout 错误。

结论：

```text
ONNX shape/layout 不是主要根因。
```

### 5.2 检查 depth history 和 warmup

C++ 初始短跑日志显示：

```text
[AutoDebug] policy run: switch to WalkNet1Action and enable InferWalkTask camera_frames=0
[VersionPolicyDebug] depth_min=0 depth_max=0 depth_mean=0
```

之后 camera worker 才显示：

```text
[AutoDebugDepth] camera_frames=37 history_size=37/37 ready=1
```

这说明 C++ policy task 刚启动时，第一帧 depth tensor 是全 0。

第一直觉是：让 AutoDebug 等 depth history 满了再启动 policy。于是做过一次实验：

```json
"DepthWarmupFrames": 37
```

结果变差。原因是：

1. 机器人在默认 target 下先站了约 37 camera frames。
2. 这段时间身体已经出现 pitch 偏移。
3. policy task 启动后，`PolicyWarmupSteps=40` 又会继续保持 default target。
4. 实际 warmup 时间比 Python 多了一段。

Python 的顺序不是“先等 depth，再启动 policy”，而是：

```text
policy task 从一开始就运行
前 40 个 policy step 不使用 actor 输出，只保持 last_action = 0
depth history 在这 40 个 policy step 中自然填充
```

因此 extra depth wait 被否定。

结论：

```text
depth 启动时序是一个可疑点，但不是主因。
不能在启用 policy task 前额外等待 depth；应该让 depth 在 policy warmup 期间填充。
```

### 5.3 对齐 root pose 和 base angular velocity

为了消除初始状态差异，C++ 加了 MuJoCo root reset：

文件：

```text
user_func.cpp
```

函数：

```cpp
ResetMujocoFreeJointLikePythonRunner(...)
```

行为：

```cpp
data->qpos[root + 0] = 0.0;
data->qpos[root + 1] = 0.0;
data->qpos[root + 2] = 0.9;
data->qpos[root + 3] = 1.0;
data->qpos[root + 4] = 0.0;
data->qpos[root + 5] = 0.0;
data->qpos[root + 6] = 0.0;
qvel root = 0;
mj_forward(model, data);
```

这与 Python 中：

```python
data.qpos[2] = root_height
data.qpos[3] = 1.0
```

一致。

base angular velocity 也改成可选使用 MuJoCo root qvel：

```json
"UseMujocoRootQvelAngVel": true
```

C++ 中对应逻辑：

```cpp
if (this->UseMujocoRootQvelAngVel) {
    this->Scheduler->template GetData<"MujocoRootAngVelValue">(AngVel);
}
```

这与 Python 的 `data.qvel[3:6]` 对齐。

结论：

```text
root pose 和 angular velocity 对齐后，仍然不能解释 C++ 行为差异。
继续排查 action/postprocess/control。
```

### 5.4 发现 action scale 顺序错误

突破点来自 C++ 启动日志。

`CommonLocoInferenceWorker` 打印：

```text
Scales_action=Vector<f,29>: [0.547546,0.547546,0.547546,0.350661,...]
```

但 bitbot 设备顺序前几个关节是：

```text
0 left_hip_pitch_joint
1 left_hip_roll_joint
2 left_hip_yaw_joint
3 left_knee_joint
4 left_ankle_pitch_joint
5 left_ankle_roll_joint
```

这些关节对应的正确 action scale 应该是：

```text
left_hip_pitch_joint: 0.547546
left_hip_roll_joint:  0.350661
left_hip_yaw_joint:   0.547546
left_knee_joint:      0.350661
left_ankle_pitch:     0.438577
left_ankle_roll:      0.438577
```

也就是设备顺序前 6 个应为：

```text
[0.547546, 0.350661, 0.547546, 0.350661, 0.438577, 0.438577]
```

而日志中是：

```text
[0.547546, 0.547546, 0.547546, 0.350661, 0.350661, 0.438577]
```

第 1 个 `left_hip_roll_joint` 明显错了。

继续看 C++ 逻辑：

```cpp
this->ActionScale[i] = PostprocessCfg["action_scale"][i].get<InferencePrecision>();
this->TrainingActionScale = DeviceToTrain(this->ActionScale);
```

这说明 JSON 中的 `action_scale` 应该先是 bitbot 设备顺序，然后 C++ 再通过 `DeviceToTrain()` 转成训练顺序。

但当前 JSON 里的数组既不是设备顺序，也不是训练顺序，因此转换后必然错。

这个错误会直接导致：

```cpp
ScaledAction = ClipedLastAction * this->TrainingActionScale + this->TrainingJointDefaultPos;
```

中的 `TrainingActionScale` 错误。actor 输出本身即使是正确的，也会被错误缩放到关节 target。

结论：

```text
这是第一个确认根因。
C++ 对 actor 输出的解释错了。
```

### 5.5 joint limit 顺序也存在同类风险

同一段逻辑中，joint clip 也是从 JSON 读：

```cpp
this->JointClipUpper[i] = joint_clip_upper_cfg[i].get<InferencePrecision>();
this->JointClipLower[i] = joint_clip_lower_cfg[i].get<InferencePrecision>();
this->TrainingJointClipUpper = DeviceToTrain(this->JointClipUpper);
this->TrainingJointClipLower = DeviceToTrain(this->JointClipLower);
```

如果 JSON 数组顺序不是设备序，clip 也会错。

错误 clip 的影响：

- 某些关节被错误限制到其他关节的 range
- actor 输出会被截断到错误范围
- 高自由度人形 gait 会迅速偏离训练分布

因此 action scale 和 joint limit 必须一起修。

### 5.6 修复 action scale / joint limit

文件：

```text
CtrlZ/CtrlZ/Workers/NN/UnitreeRlVersionInferenceWoker.hpp
```

修改方式：

不再信任 `CtrlConfig_mujoco.json` 里的旧数组顺序，而是在 version policy worker 中使用训练侧的 reference 常量：

```cpp
this->TrainingActionScale = TrainingActionScaleReference();
this->TrainingJointClipUpper = TrainingJointClipUpperReference();
this->TrainingJointClipLower = TrainingJointClipLowerReference();
```

并打印：

```cpp
std::cout << "TrainingActionScale(reference)=" << this->TrainingActionScale << std::endl;
```

训练顺序 action scale 来自 Python reference：

```text
mujoco_sim2sim/sim2sim/constants.py
```

对应训练顺序：

```text
left_shoulder_pitch_joint
right_shoulder_pitch_joint
waist_pitch_joint
left_shoulder_roll_joint
right_shoulder_roll_joint
waist_roll_joint
left_shoulder_yaw_joint
right_shoulder_yaw_joint
waist_yaw_joint
left_elbow_joint
right_elbow_joint
left_hip_pitch_joint
right_hip_pitch_joint
left_wrist_roll_joint
right_wrist_roll_joint
left_hip_roll_joint
right_hip_roll_joint
left_wrist_pitch_joint
right_wrist_pitch_joint
left_hip_yaw_joint
right_hip_yaw_joint
left_wrist_yaw_joint
right_wrist_yaw_joint
left_knee_joint
right_knee_joint
left_ankle_pitch_joint
right_ankle_pitch_joint
left_ankle_roll_joint
right_ankle_roll_joint
```

修复后日志中出现：

```text
TrainingActionScale(reference)=Vector<f,29>: [0.438577,0.438577,0.438577,...]
```

这与 Python reference 的训练顺序一致。

### 5.7 发现 KP/KD 被 ActionManager 再次错序覆盖

修复 action scale 后，机器人能明显向前走，不再早期侧倒，但还没完全达到 Python 效果。

继续看启动日志时，发现 `ActionManagementWorker` 打印的 WalkNet KP/KD 有问题。

原日志中 WalkNet 的 `PositionKp` 前几项为：

```text
[40.1792, 40.1792, 40.1792, 99.0984, 99.0984, 28.5012, ...]
```

但 bitbot 设备顺序前 6 个关节是：

```text
left_hip_pitch_joint
left_hip_roll_joint
left_hip_yaw_joint
left_knee_joint
left_ankle_pitch_joint
left_ankle_roll_joint
```

正确 KP 应该是：

```text
[40.1792, 99.0984, 40.1792, 99.0984, 28.5012, 28.5012]
```

也就是说：

```text
left_hip_roll_joint 的 KP 应该是 99.0984，但被设置成了 40.1792。
```

这会直接改变腿部控制刚度，使得 C++ 侧的低层控制不再等价于 Python PD。

进一步确认：

`settings/bitbot_mujoco.xml` 中每个 `MujocoJoint` 的 `pos_kp/pos_kd` 本来是正确的设备序；但 `ActionManagementWorker` 在切换到 `WalkNet1Action` 时，会从 `CtrlConfig_mujoco.json` 的 `ActionManager.MotorProperties` 再覆盖一遍。

因此实际运行时用的是 ActionManager 覆盖后的错误 KP/KD。

结论：

```text
这是第二个确认根因。
底层 PD 参数在 policy 切换时被错序覆盖。
```

### 5.8 修复 KP/KD 覆盖

文件：

```text
user_func.cpp
```

新增逻辑：

```cpp
OverrideMujocoMotorPropertiesToDeviceOrder(cfg_workers);
```

只在 `BUILD_SIMULATION` 下执行。

作用：

在创建 `ActionManagementWorker` 之前，把：

```json
Workers.ActionManager.MotorProperties[].Stiffness
Workers.ActionManager.MotorProperties[].Damping
```

覆盖为 bitbot 设备顺序下的正确 KP/KD。

修复后的启动日志中 WalkNet KP/KD 变为：

```text
PositionKp:
[40.1792,99.0984,40.1792,99.0984,28.5012,28.5012,...]

VelocityKd:
[2.55789,6.3088,2.55789,6.3088,1.81445,1.81445,...]
```

这与 `bitbot_mujoco.xml` 中每个 joint device 的顺序一致。

### 5.9 torque limit 对齐

之前还对 MuJoCo joint device 添加了 torque clipping。

相关文件：

```text
build/_deps/bitbot_mujoco-src/src/device/mujoco_joint.cpp
build/_deps/bitbot_mujoco-src/include/bitbot_mujoco/device/mujoco_joint.h
settings/bitbot_mujoco.xml
```

目的：

让 C++ position mode 的 PD torque 与 Python 中的：

```python
pd_torque(..., effort_limit)
```

更一致。

`bitbot_mujoco.xml` 中为每个 `MujocoJoint` 添加了：

```xml
torque_limit="..."
```

典型值：

```text
hip yaw/pitch: 88
hip roll/knee: 139
ankle/waist roll/pitch: 50
shoulder/elbow/wrist_roll: 25
wrist_pitch/wrist_yaw: 5
```

这一项不是最终主因，但属于必要对齐项。

### 5.10 heading hold 修正直行偏航

修复 action scale 和 KP/KD 后，机器人已经能稳定上台阶，但仍有一定 y 漂移。

Python 参考在 8 秒左右：

```text
y ≈ 0.16
```

C++ 修复后不开 heading hold 时，较长运行中 y 漂移可到 1m 以上。

因此启用已有内部 heading hold：

文件：

```text
settings/CtrlConfig_mujoco.json
```

配置：

```json
"HeadingHoldKp": 2.0
```

C++ 中逻辑：

```cpp
const InferencePrecision yaw_error = WrapToPi(this->HeadingHoldTargetYaw - Ang[2]);
UserCmd3[2] += this->HeadingHoldKp * yaw_error;
```

这不会改变外部接口。外部仍然只给：

```text
[vx, vy, yaw_rate] = [0.6, 0.0, 0.0]
```

内部自动加入小的 yaw rate correction，使机器人尽量保持初始 yaw。

验证中：

- `HeadingHoldKp=1.0` 时，y 漂移明显下降。
- `HeadingHoldKp=2.0` 时，y 漂移进一步降低，并未观察到提前失稳。

## 6. 最终修改点

### 6.1 `UnitreeRlVersionInferenceWoker.hpp`

路径：

```text
/home/bytedance/Documents/unitree_bitbot_hhd/CtrlZ/CtrlZ/Workers/NN/UnitreeRlVersionInferenceWoker.hpp
```

主要修改：

1. 增加使用 MuJoCo root angular velocity 的选项：

```cpp
UseMujocoRootQvelAngVel
```

2. 增加 heading hold：

```cpp
HeadingHoldKp
HeadingHoldMaxYawRate
HeadingHoldTargetYaw
```

3. version policy 使用训练序 reference action scale：

```cpp
TrainingActionScaleReference()
```

4. version policy 使用训练序 reference joint limits：

```cpp
TrainingJointClipUpperReference()
TrainingJointClipLowerReference()
```

5. 保持训练顺序与设备顺序的显式映射：

```cpp
TrainToDeviceIndex
DeviceToTrain()
TrainToDevice()
```

### 6.2 `user_func.cpp`

路径：

```text
/home/bytedance/Documents/unitree_bitbot_hhd/user_func.cpp
```

主要修改：

1. MuJoCo root reset：

```cpp
ResetMujocoFreeJointLikePythonRunner()
```

2. 缓存 MuJoCo model/data，用于 root pose log 和 qvel 读取。

3. 相机 worker 中对 depth 进行：

```text
64x36 raw depth
crop top=18, left=16
18x32
optional GaussianBlur
optional horizontal flip
normalize by 2.5
history skip = 5
```

4. MuJoCo 模式下覆盖 ActionManager 的 KP/KD：

```cpp
OverrideMujocoMotorPropertiesToDeviceOrder(cfg_workers)
```

5. AutoDebug 输出 root pose、depth stats、policy stats，方便后续对齐。

### 6.3 `settings/CtrlConfig_mujoco.json`

路径：

```text
/home/bytedance/Documents/unitree_bitbot_hhd/settings/CtrlConfig_mujoco.json
```

关键配置：

```json
"ForceDepthValue": -1.0,
"ReverseProprioHistory": false,
"UseAlterImu": false,
"FlipDepthHorizontal": true,
"DepthGaussianBlur": true,
"InvertAngVelZ": false,
"UseMujocoRootQvelAngVel": true,
"UseImuRotationMatrixProjectedGravity": true,
"HeadingHoldKp": 2.0,
"HeadingHoldMaxYawRate": 0.6,
"PolicyWarmupSteps": 40
```

### 6.4 `settings/bitbot_mujoco.xml`

路径：

```text
/home/bytedance/Documents/unitree_bitbot_hhd/settings/bitbot_mujoco.xml
```

关键点：

- 使用 `models/g1_29dof_stairs_sim2sim.xml`
- 每个 `MujocoJoint` 设置正确设备序的 `pos_kp/pos_kd`
- 每个 `MujocoJoint` 添加 `torque_limit`

### 6.5 vendored `bitbot_mujoco` joint device

路径：

```text
build/_deps/bitbot_mujoco-src/src/device/mujoco_joint.cpp
build/_deps/bitbot_mujoco-src/include/bitbot_mujoco/device/mujoco_joint.h
```

修改：

- 读取 XML attribute:

```cpp
torque_limit
```

- position/velocity/torque mode 下对 `mj_d->ctrl[mj_actuator_id_]` 进行 clamp。

## 7. 验证结果

### 7.1 编译验证

命令：

```bash
cmake --build build -j$(nproc)
```

结果：

```text
build passed
```

仅有已有 narrowing conversion warning，不影响本次问题。

### 7.2 修复前行为

修复前：

- policy 能推理
- action 有输出
- 但机器人 gait 明显错误
- 侧向漂移严重
- 不能稳定上台阶

典型日志：

```text
depth_min=0 depth_max=0 depth_mean=0
raw_action 看似正常
target_device 被错误 scale/clip 后语义错误
```

### 7.3 只修 action scale / joint limits 后

现象：

- 不再早期侧倒
- 能持续向 +X 前进
- 能开始上台阶
- 但 y 漂移仍较明显

这说明 action scale / joint limits 是主因之一。

### 7.4 再修 KP/KD 顺序后

现象明显改善。

长跑日志中，机器人能持续上台阶：

```text
policy_steps=3000 pos=2.87976,0.566025,1.79432
policy_steps=4000 pos=3.88051,1.50668,2.00385
policy_steps=5000 pos=5.00026,1.86346,1.85837
```

说明已经不是“走不明白”，而是能完成上台阶，但直行偏航还需修正。

### 7.5 HeadingHoldKp=2.0 后

验证日志：

```text
HeadingHoldKp=2
policy_steps=250  pos=0.06298,-0.00018,0.75866
policy_steps=1000 pos=0.96975,-0.07694,0.79581
policy_steps=2000 pos=2.07221,-0.00347,1.21219
policy_steps=2750 pos=2.82270,0.13147,1.75305
policy_steps=3500 pos=3.54592,0.43135,1.97815
policy_steps=3750 pos=3.86025,0.41894,1.98584
```

对比不开 heading hold：

```text
y drift > 1.0
```

启用后：

```text
y drift ≈ 0.42 at x≈3.86
```

效果明显改善。

## 8. 根因总结

### 8.1 最大根因：训练关节顺序和设备关节顺序混用

该 policy 的训练顺序不是 bitbot 设备顺序。代码里已经有：

```cpp
TrainToDeviceIndex
DeviceToTrain()
TrainToDevice()
```

说明作者已经意识到顺序不同。

但是实际配置中：

- `action_scale`
- `joint_clip_upper`
- `joint_clip_lower`
- `ActionManager.MotorProperties.Stiffness`
- `ActionManager.MotorProperties.Damping`

并没有严格保持同一种顺序。

最终导致：

```text
actor 输出的第 i 维并没有作用到训练时对应的第 i 个关节语义上。
```

高自由度人形对这种错误非常敏感。即使只有 scale/KP/KD 某几项错序，也会造成：

- 脚踝/髋部幅度不对
- 膝关节 target 不对
- 上肢和腰部补偿异常
- 身体 yaw/roll/pitch 漂移
- gait 进入训练分布外

### 8.2 depth warmup 是干扰项，不是主因

C++ 第一帧 depth 全 0 看起来很可疑，但实验表明额外等 depth 反而更差。

原因是 Python 本身也不是先等 depth 再启动 policy，而是在前 40 个 policy step 中保持 default target，同时填充 history。

所以正确方向不是“等 depth 后再启动 policy”，而是保持：

```text
policy task 正常启动
PolicyWarmupSteps=40
depth history 自然填充
```

### 8.3 KP/KD 错序会掩盖 action 修复效果

即使 action scale 修好了，如果 ActionManager 在 policy 切换时又把 KP/KD 错序覆盖，底层控制仍然和 Python 不一致。

这也是为什么修 action scale 后效果变好，但仍不够；再修 KP/KD 后才明显能上台阶。

### 8.4 C++ 与 Python depth 来源不同，仍可能带来残余差异

Python 使用 MuJoCo Renderer depth：

```python
renderer.enable_depth_rendering()
renderer.render()
```

C++ 使用 raycaster plugin sensor：

```text
depth_camera_sensor
```

单帧比较曾显示两者可以接近，但仍可能存在：

- ray direction 差异
- depth convention 差异
- startup sensor availability 差异
- discretization 差异

这可能解释修复后仍有少量轨迹差异和 yaw drift。

## 9. 当前状态

当前 C++ 已经从：

```text
同 policy 下走不明白
```

改善为：

```text
可以稳定向前并上 20cm/30cm 台阶
```

主要行为已经接近 Python sim2sim。

仍有残余差异：

- y drift 仍比 Python 大
- 过顶部平台后较长时间运行可能出现姿态下降
- C++ raycaster depth 与 Python renderer depth 不可能完全 bitwise 一致

但从目标“让 C++ 代码得到和 Python 代码一样的效果”来看，关键错误已经定位并修复，当前差异主要进入调参/细节对齐阶段。

## 10. 后续建议

### 10.1 将关节参数从 JSON 长数组改成按 joint name 配置

当前最大风险来自长数组：

```json
"action_scale": [...]
"joint_clip_upper": [...]
"joint_clip_lower": [...]
"Stiffness": [...]
"Damping": [...]
```

长数组非常容易错序，而且日志很难一眼发现。

建议改为：

```json
"ActionScaleByJointName": {
  "left_hip_pitch_joint": 0.5475464652,
  "left_hip_roll_joint": 0.3506614664
}
```

然后 C++ 根据明确的 joint order 自动生成 device-order 和 train-order vector。

这样可以彻底避免同类问题。

### 10.2 保留启动时 mapping table 和参数校验

建议 C++ 启动时打印或写入：

```text
idx
device_joint_name
train_joint_name
default_pos
action_scale
kp
kd
joint_limit
```

并自动检查：

```text
TrainToDevice(DeviceToTrain(x)) == x
```

以及关键关节参数是否符合预期。

### 10.3 对 actor input 做一次 Python/C++ 数值快照对比

当前已经通过行为验证修复了主问题，但如果要做到更严格的 sim2sim 对齐，可以进一步在同一初始状态下导出：

```text
proprio_input[768]
depth_input[1,8,18,32]
depth_latent[128]
actor_input[896]
actor_output[29]
target_pos[29]
```

然后 Python 和 C++ 做逐元素 diff。

这可以定位剩余 drift 是否来自：

- projected gravity
- qvel
- depth latent
- action target
- PD dynamics

### 10.4 保留 HeadingHold，但不要把它当成根因修复

`HeadingHoldKp=2.0` 是合理的工程修正，因为用户接口仍然保持：

```text
只给 vx
内部修正 yaw
```

但它不是替代 sim2sim 对齐的根因修复。真正的根因还是 action scale、joint limit、KP/KD 顺序。

### 10.5 不建议继续通过 ForceDepthValue 调试

之前实验：

```json
"ForceDepthValue": 1.0
```

效果更差。

说明该 policy 需要真实 depth 信息，不是单纯 blind walking policy。后续应继续保持：

```json
"ForceDepthValue": -1.0
```

## 11. 结论

这次问题的核心不是 policy 本身，也不是 ONNX Runtime 推理错误，而是 C++ 部署中多处参数顺序没有和训练/Python reference 保持一致。

最重要的两处错误是：

```text
1. actor 输出后处理的 action scale / joint limit 错序
2. policy 切换时 ActionManager 覆盖的 KP/KD 错序
```

修复后，C++ 已经能够复现 Python sim2sim 的主要效果：稳定前进并上 20cm 高、30cm 深台阶。

这也说明：

```text
policy 是可用的；
Python reference 是正确的；
C++ 原问题主要来自部署链路的 joint order 和 control parameter mismatch。
```

