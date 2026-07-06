# real_stereo_camera2sim

`real_stereo_camera2sim` 是一个从真实 RGB-D 相机录制到 MuJoCo 仿真采集的 real2sim 工作流。

当前默认配置面向 Orbbec Gemini 336L：

1. 使用 Orbbec ROS2 包启动相机并录制 RGB-D bag
2. 使用 ORB-SLAM3 RGB-D 计算真实相机轨迹
3. 将相机轨迹回放到 MuJoCo mocap body
4. 使用 `mujoco_ray_caster` 的两个 `ray_caster_camera` 采集仿真深度，并同步保存 MuJoCo RGB
5. 可视化或导出 `real_rgb / sim_rgb / real_depth / stereo raycaster / normal raycaster` 对比结果

## 目录关系

这个项目应放在 `mujoco_ray_caster` 仓库内：

```text
mujoco_ray_caster/
  raycaster_src/
  ray_plugin.cc
  ...
  real2sim/
    real_stereo_camera2sim/
```

运行命令时建议先进入项目目录：

```bash
cd /home/albusgive2/mujoco_ray_caster/real2sim/real_stereo_camera2sim
```

数据默认保存在：

```text
recordings/   # ROS2 bag
datasets/     # TUM RGB-D 数据集、ORB-SLAM3 轨迹、MuJoCo 采集结果
```

## 构建

一条命令构建：

```bash
./build.sh
```

构建脚本会做这些事：

1. 可选检查系统和 Python/conda 环境依赖
2. 构建 `mujoco_ray_caster/lib/libsensor_raycaster.so`
3. 自动拉取并编译 ORB-SLAM3 到 `third_party/ORB_SLAM3`
4. 构建本项目的 C++ MuJoCo RGB-D 采集器 `bin/mujoco_rgbd_collector`

构建脚本参数：

- `--yes`
  自动回答提示问题。
- `--skip-deps`
  不询问是否检查环境依赖。
- `--skip-orbslam`
  不拉取也不编译 ORB-SLAM3，只构建 ray caster 插件和本项目 C++ 采集器。
- `--skip-raycaster`
  不构建 `libsensor_raycaster.so`。

构建相关环境变量：

- `MJ_CONDA_ENV`
  MuJoCo Python 环境名，默认是 `mj`。
- `ORB_SLAM3_ROOT`
  ORB-SLAM3 源码目录，默认是 `third_party/ORB_SLAM3`。
- `ORB_SLAM3_REPO`
  ORB-SLAM3 git 地址，默认是 `https://github.com/UZ-SLAMLab/ORB_SLAM3.git`。
- `ORB_SLAM3_REF`
  可选的 ORB-SLAM3 branch、tag 或 commit。
- `PANGOLIN_LIB_DIR`
  ORB-SLAM3 运行时 Pangolin 库路径；如果你的 Pangolin 不在系统默认路径，可以用它补充 `LD_LIBRARY_PATH`。

如果你本机已经有可用的 ORB-SLAM3，可以这样复用：

```bash
ORB_SLAM3_ROOT=/home/albusgive2/ORB_SLAM3-1.0-release ./build.sh --skip-deps
```

只测试本项目 C++ 采集器时：

```bash
./build.sh --skip-deps --skip-orbslam
```

## 1. 启动相机节点

```bash
./launch_orbbec_gemini336l.sh
```

这个脚本会 source ROS2 Humble 和 Orbbec ROS2 工作空间，然后启动官方 Orbbec launch。

参数：

- 位置参数会透传给 `ros2 launch orbbec_camera gemini_330_series.launch.py`。
- `ROS_WS`
  Orbbec ROS2 工作空间，默认 `/home/albusgive2/deep_camear_ros`。
- `ORBBEC_PARAMS`
  Orbbec 参数文件，默认 `gemini336l_orbbec_params.yaml`。

示例：

```bash
./launch_orbbec_gemini336l.sh camera_name:=camera
```

录制 RGB-D bag：

```bash
./record_rgbd_bag.sh
```

按 `Ctrl-C` 停止录制后，bag 会保存在：

```text
recordings/gemini336l_YYYYMMDD_HHMMSS
```

`record_rgbd_bag.sh` 支持的环境变量：

- `ROS_WS`
  Orbbec ROS2 工作空间。
- `RGB_TOPIC`
  默认 `/camera/color/image_raw`。
- `DEPTH_TOPIC`
  默认 `/camera/depth/image_raw`。
- `RGB_INFO_TOPIC`
  默认 `/camera/color/camera_info`。
- `DEPTH_INFO_TOPIC`
  默认 `/camera/depth/camera_info`。

## 2. 计算真实相机轨迹

```bash
./process_latest_recording.sh
```

第一次运行时会：

1. 读取 `recordings/` 中最新的 bag
2. 导出为 ORB-SLAM3 TUM RGB-D 数据集
3. 运行 ORB-SLAM3 RGB-D
4. 保存 `CameraTrajectory.txt` 和 `KeyFrameTrajectory.txt`
5. 打开轨迹和 RGB-D 同步可视化

再次运行时，如果对应数据集已有 `CameraTrajectory.txt`，会跳过计算并直接可视化。

参数：

- 位置参数 `dataset_name`
  指定输出数据集目录名，结果写到 `datasets/<dataset_name>/`。
- `FORCE=1`
  强制重新导出并重跑 ORB-SLAM3。
- `NO_VIEWER=1`
  不打开可视化窗口。

示例：

```bash
FORCE=1 ./process_latest_recording.sh
NO_VIEWER=1 ./process_latest_recording.sh
./process_latest_recording.sh gemini336l_test
```

注意：位置参数只改变输出数据集名。这个脚本始终处理 `recordings/` 里最新的 bag。

ORB-SLAM3 路径由 `ORB_SLAM3_ROOT` 控制，默认是：

```text
third_party/ORB_SLAM3
```

## 3. 回放轨迹到 MuJoCo，并采集 sim 数据

只回放轨迹：

```bash
./replay_latest_mujoco.sh
```

指定数据集：

```bash
./replay_latest_mujoco.sh gemini336l_20260427_013045
```

回放并采集 MuJoCo 数据：

```bash
./replay_latest_mujoco.sh gemini336l_20260427_013045 --collect
```

采集结果保存在：

```text
datasets/<dataset>/mujoco_depth/
```

输出内容：

- `sim_rgb/`
  MuJoCo camera RGB，默认分辨率与真实 RGB 相同。
- `sim_depth_*`
  第一路 `ray_caster_camera`，默认配置为 `stereo_noise + baseline`。
- `secondary_depth_*`
  第二路 `ray_caster_camera`，默认配置为 `noise1 + baseline=0`。
- `frames.txt`
  帧索引、时间戳和输出文件路径。
- `collection_config.json`
  采集参数，包含深度显示范围和共享 depth clip。

`replay_latest_mujoco.sh` 通用参数：

- 位置参数 `dataset_name`
  指定数据集；不写时使用最新的带 `CameraTrajectory.txt` 的数据集。
- `--collect`
  切换到采集模式。不加就是 MuJoCo viewer 回放模式。

回放模式常用参数：

- `--xml`
  MuJoCo 场景 XML，默认 `mujoco_camera_replay.xml`。
- `--body`
  mocap body 名，默认 `tracked_camera_mocap`。
- `--camera`
  MuJoCo camera 名，默认 `tracked_camera`。
- `--fps`
  回放帧率。
- `--step`
  每次播放跳过多少个轨迹点。
- `--scale`
  轨迹平移缩放。
- `--basis`
  坐标系变换，支持 `opencv_to_mujoco`、`opencv_to_mujoco_y_back`、`ros_to_mujoco`、`identity` 或 9 个数字。
- `--origin`
  `first` 表示首帧归零，`zero` 表示保留原始轨迹原点。
- `--origin-offset "x y z"`
  在 MuJoCo 世界坐标中整体平移轨迹。
- `--z-offset`
  额外叠加 z 偏移。
- `--fixed-view`
  使用 XML 中的 `overview` 固定视角。
- `--no-loop`
  播放一遍后停止。
- `--hide-mj-camera-vis`
  不显示 MuJoCo 自带的相机可视化几何。
- `--render-dir`
  导出回放中的 MuJoCo camera RGB。
- `--render-size`
  导出分辨率，例如 `640x480`。
- `--render-only`
  只导出，不打开 viewer。
- `--dry-run`
  只检查 XML、轨迹和首帧 pose。

采集模式常用参数：

- `--xml`
  MuJoCo 场景 XML。
- `--plugin`
  `libsensor_raycaster.so` 路径，默认使用 `../../lib/libsensor_raycaster.so`。
- `--body`
  mocap body 名。
- `--camera`
  用于保存 `sim_rgb` 的 MuJoCo camera 名。
- `--sensor`
  第一路 ray caster sensor 名，默认 `tracked_depth_raycaster`。
- `--secondary-sensor`
  第二路 ray caster sensor 名，默认 `tracked_aux_depth_raycaster`。
- `--every`
  每隔多少个轨迹点采一帧。
- `--max-frames`
  最多采多少帧，`0` 表示全部。
- `--render-size`
  `sim_rgb` 分辨率；`auto` 表示优先跟真实 RGB 一致，再退回 MJCF camera resolution。
- `--render-rgb` / `--no-render-rgb`
  是否保存 `sim_rgb/`。
- `--depth-clip-min` / `--depth-clip-max`
  共享 depth clip。小于最小值或大于最大值的三路 depth 都置为 `0`。
- `--min-depth` / `--max-depth`
  深度可视化灰度映射范围。
- `--origin`、`--origin-offset`、`--basis`、`--scale`、`--z-offset`
  与回放模式相同。
- `--no-progress`
  关闭采集进度条。

示例：

```bash
./replay_latest_mujoco.sh \
  --collect \
  --depth-clip-min 0.15 \
  --depth-clip-max 1.5 \
  --origin-offset "0 0 0.05"
```

当前 C++ 采集模式已经不再支持旧的 renderer depth 参数：

```text
--render-depth
--no-render-depth
--render-depth-noise-std
--render-depth-noise-seed
--render-depth-clip-min
--render-depth-clip-max
--viewer
--viewer-fps
```

第三路 depth 现在来自第二个 `ray_caster_camera`，不是 MuJoCo renderer。

## 4. 可视化 real/sim 对比

打开窗口播放：

```bash
./play_latest_mujoco_comparison.sh
```

不开窗口，直接导出视频：

```bash
./play_latest_mujoco_comparison.sh --no-gui --save-video
```

指定输出视频路径：

```bash
./play_latest_mujoco_comparison.sh --no-gui --save-video /tmp/mj_compare.mp4
```

显示布局：

```text
real_rgb | sim_rgb | sim_rgb
real_depth | stereo_raycaster_camera | raycaster_camera
```

参数：

- 位置参数 `dataset_name`
  指定数据集；不写时使用最新数据集。
- `--sim-dir`
  指定 MuJoCo 采集目录，默认 `DATASET/mujoco_depth`。
- `--xml`
  用来读取默认 `dis_range` 的 MJCF 文件。
- `--sensor`
  用来读取 `dis_range` 的 sensor 名。
- `--depth-factor`
  真实深度 PNG 缩放因子，默认 `1000.0`。
- `--min-depth` / `--max-depth`
  三路 depth 的灰度显示范围。
- `--depth-clip-min` / `--depth-clip-max`
  覆盖 `collection_config.json` 中记录的共享 clip。
- `--depth-vis-gain`
  深度显示亮度增益，只影响可视化和导出视频，不修改原始 depth 数据。
- `--fps`
  播放或导出视频帧率。
- `--window-width`
  GUI 窗口宽度。
- `--save-video [path]`
  导出 MP4；不写路径时保存到 `DATASET/mujoco_depth_comparison.mp4`。
- `--save-first path`
  只导出第一帧对比图。
- `--no-gui`
  不打开 OpenCV 窗口。
- `--max-frames`
  最多播放或导出多少帧。
- `--no-loop`
  播放一遍后停止。

如果 depth 显示太暗，可以提高显示增益：

```bash
./play_latest_mujoco_comparison.sh --depth-vis-gain 4.0
```

或者缩小显示深度范围：

```bash
./play_latest_mujoco_comparison.sh \
  --min-depth 0.15 \
  --max-depth 1.0 \
  --depth-vis-gain 2.0
```

## MuJoCo 场景配置

主场景文件：

```text
mujoco_camera_replay.xml
mujoco_custom_scene.xml
```

`mujoco_camera_replay.xml` 中有两路 ray caster：

- `tracked_depth_raycaster`
  默认用于 stereo depth，配置 `baseline=0.095` 和 `stereo_noise`。
- `tracked_aux_depth_raycaster`
  默认用于普通 ray caster depth，配置 `baseline=0.0` 和 `noise1`。

你可以直接修改 MJCF 中的这些参数：

- `focal_length`
- `horizontal_aperture`
- `vertical_aperture`
- `size`
- `dis_range`
- `baseline`
- `lossangle`
- `min_energy`
- `noise_type`
- `noise_cfg`

修改 `dis_range` 后：

- 重新采集才会改变 MuJoCo ray caster 的真实测距范围。
- 播放器会读取新的显示范围。

## 推荐流程

```bash
./build.sh
./launch_orbbec_gemini336l.sh
./record_rgbd_bag.sh
./process_latest_recording.sh
./replay_latest_mujoco.sh --collect --depth-clip-min 0.15 --depth-clip-max 1.5
./play_latest_mujoco_comparison.sh --no-gui --save-video --depth-vis-gain 2.0
```
