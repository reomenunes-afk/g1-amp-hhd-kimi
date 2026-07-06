# Real Robot Joint Order Alignment Test

本文档说明当前 C++ 工程中仿真侧、实机侧和 policy 侧的关节顺序对应关系，以及如何使用 `system_test` 事件在 MuJoCo 和真机上做一个低风险的动作顺序验证。

## 背景

这个工程有两个目标：

1. 在 MuJoCo 中测试 sim2sim 效果。
2. 在尽量不修改接口的情况下，把同一套控制链路用于 Unitree G1 实机测试。

因此，最关键的要求是：仿真和实机的外部操作接口、内部 `MotorVec` 语义、policy 输入输出语义必须一致。

当前统一后的内部设备顺序是：

| Index | Joint |
| --- | --- |
| 0 | left_hip_pitch |
| 1 | left_hip_roll |
| 2 | left_hip_yaw |
| 3 | left_knee |
| 4 | left_ankle_pitch |
| 5 | left_ankle_roll |
| 6 | right_hip_pitch |
| 7 | right_hip_roll |
| 8 | right_hip_yaw |
| 9 | right_knee |
| 10 | right_ankle_pitch |
| 11 | right_ankle_roll |
| 12 | waist_yaw |
| 13 | waist_roll |
| 14 | waist_pitch |
| 15 | left_shoulder_pitch |
| 16 | left_shoulder_roll |
| 17 | left_shoulder_yaw |
| 18 | left_elbow |
| 19 | left_wrist_roll |
| 20 | left_wrist_pitch |
| 21 | left_wrist_yaw |
| 22 | right_shoulder_pitch |
| 23 | right_shoulder_roll |
| 24 | right_shoulder_yaw |
| 25 | right_elbow |
| 26 | right_wrist_roll |
| 27 | right_wrist_pitch |
| 28 | right_wrist_yaw |

## 已对齐的内容

### 1. 仿真和实机 XML 设备顺序一致

MuJoCo 配置：

```text
settings/bitbot_mujoco.xml
```

实机配置：

```text
settings/bitbot_unitree.xml
```

两者的 0-28 号关节都使用同一个设备顺序：

```text
left leg -> right leg -> waist -> left arm -> right arm
```

### 2. `JOINT_ID_MAP` 已统一

之前实机侧 `JOINT_ID_MAP` 是：

```cpp
0, 6, 12, 1, 7, 13, ...
```

这会把 `d.JointsPtr` 内部顺序打乱。由于 `UnitreeBus` 本身已经按 XML 顺序读写 `motor_state()[i]` / `motor_cmd()[i]`，这个重排是不应该存在的。

现在仿真和实机都是：

```cpp
0, 1, 2, ..., 28
```

位置：

```text
types.hpp
```

### 3. ActionManager 的 KP/KD 已统一为设备顺序

`ActionManager.MotorProperties` 会在切换 policy 时写入：

```text
TargetMotorStiffness
TargetMotorDamping
```

然后 `MotorWorker` 在每个控制周期下发到电机。

之前 MuJoCo 侧靠 runtime override 修正 KP/KD 顺序，但实机侧不会走这个 override，因此真机仍可能错序。现在 `settings/CtrlConfig.json` 中 `WalkNet1` 和 `DanceNet1` 的 KP/KD 已直接改为设备顺序，仿真和实机共用同一份配置。

## Policy 顺序

Policy 的训练关节顺序不是设备顺序。当前 `UnitreeRlVersionInferenceWorker` 内部显式维护：

```cpp
TrainToDeviceIndex
DeviceToTrain()
TrainToDevice()
```

数据流为：

```text
CurrentMotorPosition(device order)
  -> DeviceToTrain()
  -> policy observation(training order)

policy action(training order)
  -> TrainToDevice()
  -> TargetMotorPosition(device order)
```

因此，只要仿真和实机的 `MotorVec` 都保持同一个 device order，policy 就能在两边使用同一套映射。

## System Test 设计

新增的 `system_test` 不是 policy 测试，而是底层动作顺序测试。

触发方式：

```text
t
```

进入 `system_test` 后，代码会：

1. 停止 `ResetTask`
2. 停止 `InferWalkTask`
3. 停止 `DanceInferTask`
4. 阻塞 `ActionManager` 输出
5. 记录当前 29 个关节位置作为 baseline
6. 依次对 0-28 号关节施加一个很小的正弦偏移
7. 每次只动一个关节
8. 测试结束后回到 baseline 并保持

轨迹幅度：

```text
腿部关节: 0.06 rad
腰部关节: 0.05 rad
手臂关节: 0.10 rad
```

每个关节测试时间：

```text
2.0 s
```

初始保持时间：

```text
1.0 s
```

## 推荐测试流程

### MuJoCo 侧

启动程序后：

```text
9 -> power_on
8 -> start
t -> system_test
```

观察 MuJoCo 中每一步运动的关节，并对照终端日志：

```text
[SystemTest] active_index=0 joint=left_hip_pitch amplitude_rad=0.06
[SystemTestSample] index=0 joint=left_hip_pitch offset_rad=...
```

确认视觉上动的关节和日志中的 `joint` 一致。

### 真机侧

安全要求：

1. 真机必须吊装、支撑或处于不会跌倒的安全测试架上。
2. 不要让机器人自由站立执行完整 29 关节顺序测试。
3. 急停必须可用。
4. 第一次测试建议只观察前几个关节，确认幅度和方向正常后再继续。

建议流程：

```text
9 -> power_on
8 -> start
t -> system_test
```

如果想从 init pose 后开始：

```text
9 -> power_on
8 -> start
p -> init_pose
t -> system_test
```

测试结束后：

```text
p -> 回到 init_pose
```

或直接急停/stop。

## 如何判断顺序是否正确

每一步只应该有日志中对应的关节发生明显运动。

例如：

```text
[SystemTest] active_index=3 joint=left_knee
```

预期现象：

```text
只有左膝附近发生小幅周期运动
```

如果看到右腿、腰部、手臂或其他关节运动，说明以下链路中至少有一处仍然错序：

```text
bitbot_unitree.xml device id
JOINT_ID_MAP
Unitree SDK motor_state/motor_cmd index
MotorVec 内部顺序
ActionManager / MotorWorker 下发顺序
```

## 注意事项

`system_test` 只验证底层动作顺序，不验证以下内容：

1. IMU 坐标系是否和训练一致。
2. 深度图方向、裁剪、尺度是否和训练一致。
3. policy 在实机上的稳定性。
4. 电机实际力矩能力、延迟、摩擦和仿真是否一致。

因此，通过 `system_test` 只能说明“动作顺序基本对应”。上 policy 前仍然需要单独确认 IMU 和深度图。

