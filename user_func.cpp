/**
 * @file user_func.cpp
 * @author Zishun Zhou
 * @brief
 * @date 2025-03-10
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "user_func.h"

#include <algorithm>
#include <chrono>
#include <deque>
#define _USE_MATH_DEFINES
#include <cmath>
#include <ctime>
#include <memory>
#include <thread>
#include <iostream> // std::cout
#include <nlohmann/json.hpp>
#include <fstream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "types.hpp"

namespace
{
template <typename Vec>
void ComputeDepthStats(const Vec& values, RealNumber& min_v, RealNumber& max_v, RealNumber& mean_v)
{
    min_v = values[0];
    max_v = values[0];
    double sum = 0.0;
    for (const auto value : values) {
        min_v = std::min(min_v, static_cast<RealNumber>(value));
        max_v = std::max(max_v, static_cast<RealNumber>(value));
        sum += static_cast<double>(value);
    }
    mean_v = static_cast<RealNumber>(sum / static_cast<double>(values.size()));
}

template <typename Vec>
void PrintVecStats(const char* name, const Vec& values)
{
    RealNumber min_v = values[0];
    RealNumber max_v = values[0];
    double sum = 0.0;
    for (const auto value : values) {
        min_v = std::min(min_v, static_cast<RealNumber>(value));
        max_v = std::max(max_v, static_cast<RealNumber>(value));
        sum += static_cast<double>(value);
    }
    std::cout << " " << name << "_min=" << min_v
              << " " << name << "_max=" << max_v
              << " " << name << "_mean=" << static_cast<RealNumber>(sum / static_cast<double>(values.size()));
}

template <typename Vec>
void PrintVecHead(const char* name, const Vec& values, size_t count)
{
    std::cout << " " << name << "=[";
    const size_t n = std::min(count, values.size());
    for (size_t i = 0; i < n; ++i) {
        std::cout << values[i] << (i + 1 < n ? "," : "");
    }
    std::cout << "]";
}

constexpr std::array<const char*, JOINT_NUMBER> kDeviceOrderJointNames = {
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw"
};

RealNumber SystemTestAmplitude(size_t joint_idx)
{
    if (joint_idx < 12) {
        return static_cast<RealNumber>(0.06);
    }
    if (joint_idx < 15) {
        return static_cast<RealNumber>(0.05);
    }
    return static_cast<RealNumber>(0.10);
}

#ifdef BUILD_SIMULATION
void ResetMujocoFreeJointLikePythonRunner(const mjModel* model, mjData* data)
{
    if (model == nullptr || data == nullptr) {
        return;
    }

    const int free_joint_id = mj_name2id(model, mjOBJ_JOINT, "floating_base_joint");
    if (free_joint_id < 0) {
        std::cerr << "[MujocoInit] floating_base_joint not found; skip root reset" << std::endl;
        return;
    }

    const int qpos_adr = model->jnt_qposadr[free_joint_id];
    const int qvel_adr = model->jnt_dofadr[free_joint_id];

    // Match mujoco_sim2sim/sim2sim/robot.py::write_default_pose.
    data->qpos[qpos_adr + 0] = 0.0;
    data->qpos[qpos_adr + 1] = 0.0;
    data->qpos[qpos_adr + 2] = 0.9;
    data->qpos[qpos_adr + 3] = 1.0;
    data->qpos[qpos_adr + 4] = 0.0;
    data->qpos[qpos_adr + 5] = 0.0;
    data->qpos[qpos_adr + 6] = 0.0;

    for (int i = 0; i < 6; ++i) {
        data->qvel[qvel_adr + i] = 0.0;
    }
    if (model->nu > 0) {
        mju_zero(data->ctrl, model->nu);
    }
    mj_forward(model, data);

    std::cout << "[MujocoInit] root reset to Python sim2sim pose: xyz="
              << data->qpos[qpos_adr + 0] << ","
              << data->qpos[qpos_adr + 1] << ","
              << data->qpos[qpos_adr + 2]
              << " quat="
              << data->qpos[qpos_adr + 3] << ","
              << data->qpos[qpos_adr + 4] << ","
              << data->qpos[qpos_adr + 5] << ","
              << data->qpos[qpos_adr + 6]
              << std::endl;
}
#endif
}

 // 辅助函数：对容器中的每个元素应用函数
template <typename Container, typename Func>
void Apply(Container& container, Func func)
{
    for (size_t i = 0; i < container.size(); ++i)
    {
        func(&container[i], i);
    }
}

#ifdef BUILD_SIMULATION
void ApplyMujocoRuntimeOverrides(nlohmann::json& cfg_root, nlohmann::json& cfg_workers)
{
    if (!cfg_root.contains("AutoDebug")) {
        cfg_root["AutoDebug"] = nlohmann::json::object();
    }
    if (!cfg_root.contains("Diagnostics")) {
        cfg_root["Diagnostics"] = nlohmann::json::object();
    }

    // Keep the public config shared with the real robot. Only override values that are
    // genuinely MuJoCo-specific at runtime.
    auto& preprocess = cfg_workers["WalkNet1"]["Preprocess"];
    preprocess["UseMujocoRootQvelAngVel"] = true;
    preprocess["HeadingHoldKp"] = 2.0;
    preprocess["HeadingHoldCommandThreshold"] = 0.05;
}
#endif



void ConfigFunc(const KernelBus& bus, UserData& d)
{
    //读取json配置文件,并初始化各个worker
    nlohmann::json cfg_root;
    nlohmann::json cfg_workers;
    {
        //NOTE: 注意将配置文件路径修改为自己的路径
        std::string path = PROJECT_ROOT_DIR + std::string("/settings/CtrlConfig.json");
        std::ifstream cfg_file(path);
        cfg_root = nlohmann::json::parse(cfg_file, nullptr, true, true);
        cfg_workers = cfg_root["Workers"];
    }
#ifdef BUILD_SIMULATION
    ApplyMujocoRuntimeOverrides(cfg_root, cfg_workers);
    d.AutoDebugEnabled = cfg_root.contains("AutoDebug") && cfg_root["AutoDebug"].value("Enable", false);
    d.SimDiagnosticsEnabled = d.AutoDebugEnabled || cfg_root.value("Diagnostics", nlohmann::json::object()).value("Enable", false);
#endif

    bool flip_depth_horizontal = false;
    bool depth_gaussian_blur = true;
    if (cfg_workers.contains("WalkNet1") && cfg_workers["WalkNet1"].contains("Preprocess")) {
        const auto& preprocess_cfg = cfg_workers["WalkNet1"]["Preprocess"];
        flip_depth_horizontal = preprocess_cfg.value("FlipDepthHorizontal", false);
        depth_gaussian_blur = preprocess_cfg.value("DepthGaussianBlur", true);
    }
    std::cout << "[Config] FlipDepthHorizontal=" << flip_depth_horizontal << std::endl;
    std::cout << "[Config] DepthGaussianBlur=" << depth_gaussian_blur << std::endl;

    //获取设备指针
    d.ImuPtr = bus.GetDevice<DeviceImu>(IMU_ID_MAP).value();
    d.ImuAlterPtr = bus.GetDevice<DeviceImu>(ALTER_IMU_ID_MAP).value();
    Apply(d.JointsPtr, [&bus](DeviceJoint** joint, size_t i)
        { *joint = bus.GetDevice<DeviceJoint>(JOINT_ID_MAP[i]).value(); });

    auto camera_opt = bus.GetDevice<DeviceCamera>(CAMERA_ID_MAP);
    if (camera_opt.has_value()) {
        d.CameraPtr = camera_opt.value();
        std::cout << "Camera device initialized successfully" << std::endl;
    } else {
        d.CameraPtr = nullptr;
        std::cout << "Warning: Camera device not found" << std::endl;
    }

#ifdef BUILD_SIMULATION
    // Cache MuJoCo model/data handles so state callbacks can log root pose.
    d.MujocoModel = bus.GetMujocoModel();
    d.MujocoData = bus.GetMujocoData();
    if (d.MujocoModel != nullptr) {
        d.PelvisBodyId = mj_name2id(d.MujocoModel, mjOBJ_BODY, "pelvis");
    }
    ResetMujocoFreeJointLikePythonRunner(d.MujocoModel, d.MujocoData);
#endif

    //创建调度器
    d.TaskScheduler = SchedulerType::Create(cfg_root["Scheduler"]);

    //初始化各个worker
    d.ImuWorker = d.TaskScheduler->template CreateWorker<ImuWorkerType>(d.ImuPtr, cfg_workers["ImuProcess"]);
    d.MotorWorker = d.TaskScheduler->template CreateWorker<MotorWorkerType>(cfg_workers["MotorControl"], d.JointsPtr);
    d.Logger = d.TaskScheduler->template CreateWorker<LoggerWorkerType>(cfg_workers["AsyncLogger"]);
    d.ActionManagementWorker = d.TaskScheduler->template CreateWorker<ActionManagementWorkerType>(cfg_workers["ActionManager"]);

    d.AlterImuWorker = d.TaskScheduler->template CreateWorker<AlterImuWorkerType>([&d](SchedulerType::Ptr scheduler) {
        RealNumber roll = d.ImuAlterPtr->GetRoll();
        RealNumber pitch = d.ImuAlterPtr->GetPitch();
        RealNumber yaw = d.ImuAlterPtr->GetYaw();
        Vec3 euler_angles = Vec3({ roll, pitch, yaw });
        Vec3 gyro = Vec3({
            d.ImuAlterPtr->GetGyroX(),
            d.ImuAlterPtr->GetGyroY(),
            d.ImuAlterPtr->GetGyroZ()
        });
        scheduler->template SetData<"AlterAngleValue">(euler_angles);
        scheduler->template SetData<"AlterAngleVelocityValue">(gyro);
#ifdef BUILD_SIMULATION
        auto primary_rot_array = d.ImuPtr->GetRotationMatrix();
        auto alter_rot_array = d.ImuAlterPtr->GetRotationMatrix();
        Vec9 primary_rot;
        Vec9 alter_rot;
        for (size_t i = 0; i < 9; ++i) {
            primary_rot[i] = static_cast<RealNumber>(primary_rot_array[i]);
            alter_rot[i] = static_cast<RealNumber>(alter_rot_array[i]);
        }
        scheduler->template SetData<"RotationMatrixValue">(primary_rot);
        scheduler->template SetData<"AlterRotationMatrixValue">(alter_rot);
        if (d.MujocoData != nullptr) {
            Vec3 root_ang_vel = Vec3({
                static_cast<RealNumber>(d.MujocoData->qvel[3]),
                static_cast<RealNumber>(d.MujocoData->qvel[4]),
                static_cast<RealNumber>(d.MujocoData->qvel[5])
            });
            scheduler->template SetData<"MujocoRootAngVelValue">(root_ang_vel);
        }
#endif
    });

// 创建相机 Worker（和 AlterImuWorker 一样使用 SimpleCallbackWorker 模式）
// std::cout << "[camera] CameraWorker created" << std::endl;
    d.CameraWorker = d.TaskScheduler->template CreateWorker<AlterImuWorkerType>([&d, flip_depth_horizontal, depth_gaussian_blur](SchedulerType::Ptr scheduler) {
    if (d.CameraPtr && d.CameraPtr->IsReady()) {
        d.CameraPtr->UpdateFrame();

        auto depth_frame_opt = d.CameraPtr->GetDepthFrame();
        if (!depth_frame_opt.has_value()) {
            return;
        }

        DepthFrameType depth_frame = depth_frame_opt.value();

        float center_dist = depth_frame.get_distance(
            d.CameraPtr->GetWidth() / 2,
            d.CameraPtr->GetHeight() / 2);
        scheduler->template SetData<"DepthCenterDistance">(center_dist);

        float min_dist = 10.0f, max_dist = 0.0f;
        for (int y = 0; y < d.CameraPtr->GetHeight(); y += 10) {
            for (int x = 0; x < d.CameraPtr->GetWidth(); x += 10) {
                float dist = depth_frame.get_distance(x, y);
                if (dist > 0 && dist < 10.0f) {
                    min_dist = std::min(min_dist, dist);
                    max_dist = std::max(max_dist, dist);
                }
            }
        }

        scheduler->template SetData<"DepthMinDistance">(min_dist);
        scheduler->template SetData<"DepthMaxDistance">(max_dist);

          using ParkourDepthFrame = std::array<RealNumber, PARKOUR_DEPTH_HEIGHT * PARKOUR_DEPTH_WIDTH>;
          static std::deque<ParkourDepthFrame> parkour_depth_history;
          constexpr size_t kHistorySkipFrames = 5;
          constexpr size_t kHistoryOffset = 1;
          constexpr size_t kRawWidth = 64;
          constexpr size_t kRawHeight = 36;
          constexpr size_t kCropUp = 18;
          constexpr size_t kCropLeft = 16;
          constexpr RealNumber kDepthNormMax = 2.5;

          ParkourDepthFrame frame{};
          if (static_cast<size_t>(d.CameraPtr->GetWidth()) == kRawWidth &&
              static_cast<size_t>(d.CameraPtr->GetHeight()) == kRawHeight) {
              for (size_t y = 0; y < PARKOUR_DEPTH_HEIGHT; ++y) {
                  for (size_t x = 0; x < PARKOUR_DEPTH_WIDTH; ++x) {
                      float depth = depth_frame.get_distance(
                          static_cast<int>(x + kCropLeft),
                          static_cast<int>(y + kCropUp));
                      RealNumber normalized = 1.0;
                      if (depth > 0.0f) {
                          normalized = std::clamp(static_cast<RealNumber>(depth) / kDepthNormMax,
                                                  static_cast<RealNumber>(0.0),
                                                  static_cast<RealNumber>(1.0));
                      }
                      const size_t out_x = flip_depth_horizontal
                          ? (PARKOUR_DEPTH_WIDTH - 1 - x)
                          : x;
                      frame[y * PARKOUR_DEPTH_WIDTH + out_x] = normalized;
                  }
              }

              if (depth_gaussian_blur) {
                  ParkourDepthFrame blurred = frame;
                  constexpr RealNumber k0 = static_cast<RealNumber>(0.27406862);
                  constexpr RealNumber k1 = static_cast<RealNumber>(0.45186276);
                  const RealNumber kernel[3] = {k0, k1, k0};
                  for (size_t y = 0; y < PARKOUR_DEPTH_HEIGHT; ++y) {
                      for (size_t x = 0; x < PARKOUR_DEPTH_WIDTH; ++x) {
                          RealNumber sum = 0;
                          RealNumber weight_sum = 0;
                          for (int dy = -1; dy <= 1; ++dy) {
                              const int yy = static_cast<int>(y) + dy;
                              if (yy < 0 || yy >= static_cast<int>(PARKOUR_DEPTH_HEIGHT)) {
                                  continue;
                              }
                              for (int dx = -1; dx <= 1; ++dx) {
                                  const int xx = static_cast<int>(x) + dx;
                                  if (xx < 0 || xx >= static_cast<int>(PARKOUR_DEPTH_WIDTH)) {
                                      continue;
                                  }
                                  const RealNumber weight = kernel[dy + 1] * kernel[dx + 1];
                                  sum += frame[static_cast<size_t>(yy) * PARKOUR_DEPTH_WIDTH + static_cast<size_t>(xx)] * weight;
                                  weight_sum += weight;
                              }
                          }
                          blurred[y * PARKOUR_DEPTH_WIDTH + x] = sum / weight_sum;
                      }
                  }
                  frame = blurred;
              }

              const size_t required_history = (PARKOUR_DEPTH_HISTORY - 1) * kHistorySkipFrames + 1 + kHistoryOffset;
              if (parkour_depth_history.empty()) {
                  ParkourDepthFrame initial_frame{};
                  initial_frame.fill(static_cast<RealNumber>(1.0));
                  for (size_t i = 0; i < required_history; ++i) {
                      parkour_depth_history.push_back(initial_frame);
                  }
              }
              parkour_depth_history.push_back(frame);
              while (parkour_depth_history.size() > required_history) {
                  parkour_depth_history.pop_front();
              }
              ++d.AutoDebugCameraFrames;
              d.AutoDebugDepthHistorySize = parkour_depth_history.size();

              ParkourDepthImageVec depth_obs;
              if (parkour_depth_history.size() >= required_history) {
                  for (size_t h = 0; h < PARKOUR_DEPTH_HISTORY; ++h) {
                      const size_t history_index = parkour_depth_history.size() - required_history + kHistoryOffset + h * kHistorySkipFrames;
                      const auto& selected_frame = parkour_depth_history[history_index];
                      std::copy(selected_frame.begin(),
                                selected_frame.end(),
                                depth_obs.begin() + static_cast<std::ptrdiff_t>(h * selected_frame.size()));
                  }
              } else {
                  for (size_t h = 0; h < PARKOUR_DEPTH_HISTORY; ++h) {
                      std::copy(frame.begin(),
                                frame.end(),
                                depth_obs.begin() + static_cast<std::ptrdiff_t>(h * frame.size()));
                  }
              }
              ComputeDepthStats(depth_obs, d.AutoDebugDepthMin, d.AutoDebugDepthMax, d.AutoDebugDepthMean);
              d.AutoDebugDepthReady = parkour_depth_history.size() >= required_history;
              if (d.AutoDebugCameraFrames == 1 ||
                  d.AutoDebugCameraFrames == required_history ||
                  d.AutoDebugCameraFrames % 30 == 0) {
                  if (d.SimDiagnosticsEnabled) {
                      std::cout << "[SimDepth] camera_frames=" << d.AutoDebugCameraFrames
                                << " history_size=" << d.AutoDebugDepthHistorySize
                                << "/" << required_history
                                << " ready=" << d.AutoDebugDepthReady
                                << " min=" << d.AutoDebugDepthMin
                                << " max=" << d.AutoDebugDepthMax
                                << " mean=" << d.AutoDebugDepthMean
                                << std::endl;
                  }
              }
              scheduler->template SetData<"ParkourDepthImage">(depth_obs);

              if (d.SimDiagnosticsEnabled && d.AutoDebugCameraFrames % 30 == 0) {
                  cv::Mat raw_gray(static_cast<int>(kRawHeight), static_cast<int>(kRawWidth), CV_8UC1);
                  for (size_t y = 0; y < kRawHeight; ++y) {
                      for (size_t x = 0; x < kRawWidth; ++x) {
                          float depth = depth_frame.get_distance(static_cast<int>(x), static_cast<int>(y));
                          float norm = 1.0f;
                          if (depth > 0.0f) {
                              norm = 1.0f - std::clamp(depth / static_cast<float>(kDepthNormMax), 0.0f, 1.0f);
                          }
                          raw_gray.at<unsigned char>(static_cast<int>(y), static_cast<int>(x)) = static_cast<unsigned char>(norm * 255.0f);
                      }
                  }
                  std::string raw_path = "/tmp/mujoco_depth_raw_" + std::to_string(d.AutoDebugCameraFrames) + ".png";
                  cv::imwrite(raw_path, raw_gray);

                  cv::Mat crop_gray(static_cast<int>(PARKOUR_DEPTH_HEIGHT), static_cast<int>(PARKOUR_DEPTH_WIDTH), CV_8UC1);
                  for (size_t y = 0; y < PARKOUR_DEPTH_HEIGHT; ++y) {
                      for (size_t x = 0; x < PARKOUR_DEPTH_WIDTH; ++x) {
                          float norm = 1.0f - frame[y * PARKOUR_DEPTH_WIDTH + x];
                          crop_gray.at<unsigned char>(static_cast<int>(y), static_cast<int>(x)) = static_cast<unsigned char>(norm * 255.0f);
                      }
                  }
                  std::string crop_path = "/tmp/mujoco_depth_crop_" + std::to_string(d.AutoDebugCameraFrames) + ".png";
                  cv::imwrite(crop_path, crop_gray);
              }
          }
    }
});

    //创建主任务列表，并添加worker
    d.TaskScheduler->CreateTaskList("MainTask", 1, true);
    d.TaskScheduler->AddWorkers("MainTask",
        {
            d.ImuWorker,
            d.AlterImuWorker,
            d.MotorWorker,
        });

    const double scheduler_dt = cfg_root["Scheduler"]["dt"].get<double>();
    const double camera_fps = d.CameraPtr ? static_cast<double>(d.CameraPtr->GetFps()) : 30.0;
    const size_t camera_task_div = std::max<size_t>(
        1, static_cast<size_t>(std::llround(1.0 / (scheduler_dt * camera_fps))));

    // Match the Unitree camera model: sample depth at camera FPS in a non-main task.
    d.TaskScheduler->CreateTaskList("CameraTask", camera_task_div);
    d.TaskScheduler->AddWorker("CameraTask", d.CameraWorker);
    d.TaskScheduler->EnableTaskList("CameraTask");

    //创建推理任务列表，并添加worker，设置推理任务频率
    d.DanceNetInferWorker = d.TaskScheduler->template CreateWorker<BeyondMimicUnitreeInferWorkerType>(cfg_workers["DanceNet1"], cfg_workers["DanceNet1"], JOINT_ID_MAP);
    d.TaskScheduler->CreateTaskList("DanceInferTask", cfg_root["Scheduler"]["InferTask"]["PolicyFrequency"]);
    d.TaskScheduler->AddWorkers("DanceInferTask",
        {
            d.DanceNetInferWorker,
            d.ActionManagementWorker//,
            //d.Logger
        });

    //创建走路的推理任务列表，并添加worker，设置推理任务频率
    d.WalkNetInferWorker = d.TaskScheduler->template CreateWorker<UnitreeRlLabVelocityInferWorkerType>(cfg_workers["WalkNet1"], cfg_workers["MotorControl"]);
    d.WalkCmdWorker = d.TaskScheduler->template CreateWorker<CmdWorkerType>(cfg_workers["WalkCmd"]);
    d.TaskScheduler->CreateTaskList("InferWalkTask", cfg_root["Scheduler"]["InferTask"]["PolicyFrequency"]);
    d.TaskScheduler->AddWorkers("InferWalkTask",
        {
            d.WalkCmdWorker,
            d.WalkNetInferWorker,
            d.ActionManagementWorker,
            d.Logger
        });

    //创建复位任务列表，并添加worker，设置复位任务频率为主任务频率的1/10
    d.MotorResetWorker = d.TaskScheduler->template CreateWorker<MotorResetWorkerType>(cfg_workers["MotorControl"], cfg_workers["ResetPosition"]);
    d.TaskScheduler->CreateTaskList("ResetTask", 10);
    d.TaskScheduler->AddWorker("ResetTask", d.MotorResetWorker);

#ifdef BUILD_SIMULATION
      if (d.AutoDebugEnabled) {
          const RealNumber reset_duration = cfg_root["AutoDebug"].value("ResetDuration", static_cast<RealNumber>(3.2));
          const size_t reset_steps = static_cast<size_t>(std::llround(reset_duration / scheduler_dt));
          const size_t required_depth_history = 0;

          std::cout << "[AutoDebug] armed: wait for first valid bus read, depth warmup, then switch to WalkNet1Action. "
                    << "reset_duration=" << reset_duration
                    << "s depth_warmup_frames="
                    << cfg_root["AutoDebug"].value("DepthWarmupFrames", required_depth_history)
                    << " use_reset=" << cfg_root["AutoDebug"].value("UseReset", false)
                    << ". WalkNet1 loads "
                    << cfg_workers["WalkNet1"]["Network"]["ModelPath"].get<std::string>()
                    << std::endl;

          d.AutoDebugUseReset = cfg_root["AutoDebug"].value("UseReset", false);
          d.AutoDebugResetSteps = reset_steps;
          d.AutoDebugDelaySteps = reset_steps;
          d.AutoDebugDepthWarmupFrames = cfg_root["AutoDebug"].value("DepthWarmupFrames", required_depth_history);
      }
#endif

    //开始调度器
    d.TaskScheduler->Start();
}

void FinishFunc(UserData& d)
{
    std::cout << "Finishing user functions..." << std::endl;
    std::cout << "Goodbye!" << std::endl;
}

void StateWaitingFunc(const bitbot::KernelInterface& kernel,
    bitbot::ExtraData& extra_data, UserData& d)
{
      static bool waiting_enter_logged = false;
      if (d.SimDiagnosticsEnabled && !waiting_enter_logged) {
          std::cout << "[SimDiagnostics] StateWaitingFunc entered" << std::endl;
          waiting_enter_logged = true;
      }
      if (d.AutoDebugEnabled && !d.AutoDebugBusPrimed) {
          d.TaskScheduler->SpinOnce();
          d.MotorWorker->SetCurrentPositionAsTargetPosition();
          d.AutoDebugBusPrimed = true;
          std::cout << "[AutoDebug] bus primed with first device read before reset/policy" << std::endl;
          return;
      }
      if (d.AutoDebugEnabled && !d.AutoDebugStarted) {
          std::cout << "[AutoDebug] warmup starts after valid bus read"
                    << " use_reset=" << d.AutoDebugUseReset
                    << " reset_steps=" << d.AutoDebugResetSteps
                    << " depth_warmup_frames=" << d.AutoDebugDepthWarmupFrames
                    << std::endl;
          d.AutoDebugStarted = true;
          d.AutoDebugWaitSteps = 0;
          if (d.AutoDebugUseReset) {
              d.AutoDebugResetActive = true;
              d.MotorResetWorker->StartReset();
              d.TaskScheduler->EnableTaskList("ResetTask");
          }
      }
      if (d.AutoDebugEnabled && d.AutoDebugStarted && !d.AutoDebugPolicySwitchDone) {
          ++d.AutoDebugWaitSteps;
          const bool reset_ready = !d.AutoDebugUseReset || d.AutoDebugWaitSteps >= d.AutoDebugResetSteps;
          const bool depth_ready = d.AutoDebugCameraFrames >= d.AutoDebugDepthWarmupFrames;
          if (d.AutoDebugWaitSteps == 1 || d.AutoDebugWaitSteps % 250 == 0 || depth_ready) {
              std::cout << "[AutoDebug] waiting"
                        << " steps=" << d.AutoDebugWaitSteps
                        << " reset_ready=" << reset_ready
                        << " camera_frames=" << d.AutoDebugCameraFrames
                        << "/" << d.AutoDebugDepthWarmupFrames
                        << " depth_ready=" << depth_ready
                        << " depth_min=" << d.AutoDebugDepthMin
                        << " depth_max=" << d.AutoDebugDepthMax
                        << " depth_mean=" << d.AutoDebugDepthMean
                        << std::endl;
          }
          if (d.AutoDebugWaitSteps == 1 || d.AutoDebugWaitSteps % 250 == 0) {
              Vec3 angle;
              Vec3 ang_vel;
              MotorVec current_pos;
              MotorVec target_pos;
              d.TaskScheduler->template GetData<"AngleValue">(angle);
              d.TaskScheduler->template GetData<"AngleVelocityValue">(ang_vel);
              d.TaskScheduler->template GetData<"CurrentMotorPosition">(current_pos);
              d.TaskScheduler->template GetData<"TargetMotorPosition">(target_pos);
              std::cout << "[AutoDebugState] steps=" << d.AutoDebugWaitSteps
                        << " angle=" << angle
                        << " ang_vel=" << ang_vel;
              PrintVecStats("current_pos", current_pos);
              PrintVecStats("target_pos", target_pos);
              PrintVecHead("current_head", current_pos, 12);
              PrintVecHead("target_head", target_pos, 12);
              std::cout << std::endl;
          }
          if (reset_ready && depth_ready) {
              std::cout << "[AutoDebug] policy run: switch to WalkNet1Action and enable InferWalkTask"
                        << " camera_frames=" << d.AutoDebugCameraFrames
                        << " depth_ready=" << d.AutoDebugDepthReady
                        << " depth_min=" << d.AutoDebugDepthMin
                        << " depth_max=" << d.AutoDebugDepthMax
                        << " depth_mean=" << d.AutoDebugDepthMean
                        << std::endl;
              d.MotorResetWorker->StopReset();
              d.TaskScheduler->DisableTaskList("ResetTask");
              d.ActionManagementWorker->SwitchTo("WalkNet1Action", 0.0f);
              d.TaskScheduler->EnableTaskList("InferWalkTask");
              d.TaskScheduler->DisableTaskList("DanceInferTask");
              d.AutoDebugResetActive = false;
              d.AutoDebugPolicyActive = true;
              d.AutoDebugPolicySwitchDone = true;
          }
          d.TaskScheduler->SpinOnce();
          return;
      }
      if (d.AutoDebugPolicyActive) {
#ifdef BUILD_SIMULATION
          static size_t policy_run_steps = 0;
          ++policy_run_steps;
          if (d.MujocoData != nullptr && d.PelvisBodyId >= 0 && policy_run_steps % 250 == 0)
          {
              const mjData* mj_d = d.MujocoData;
              const int b = d.PelvisBodyId;
              std::cout << "[AutoDebugRoot] policy_steps=" << policy_run_steps
                        << " pos=" << mj_d->xpos[3 * b] << ","
                        << mj_d->xpos[3 * b + 1] << ","
                        << mj_d->xpos[3 * b + 2]
                        << " quat=" << mj_d->xquat[4 * b + 0] << ","
                        << mj_d->xquat[4 * b + 1] << ","
                        << mj_d->xquat[4 * b + 2] << ","
                        << mj_d->xquat[4 * b + 3]
                        << std::endl;
          }
#endif
          d.TaskScheduler->SpinOnce();
          return;
      }
      if (d.AutoDebugResetActive) {
          d.TaskScheduler->SpinOnce();
          return;
      }
    //空闲等待状态，重置目标位置防止突变
    d.MotorWorker->SetCurrentPositionAsTargetPosition();
    //调度器进行一次调度
    d.TaskScheduler->SpinOnce();
}

void StateSystemTestFunc(const bitbot::KernelInterface& kernel,
    bitbot::ExtraData& extra_data, UserData& d)
{
    constexpr RealNumber kPi = static_cast<RealNumber>(3.14159265358979323846);
    constexpr RealNumber kJointSegmentSeconds = static_cast<RealNumber>(2.0);
    constexpr RealNumber kInitialHoldSeconds = static_cast<RealNumber>(1.0);

    if (!d.SystemTestInitialized) {
        d.TaskScheduler->template GetData<"CurrentMotorPosition">(d.SystemTestBaseline);
        d.SystemTestStep = 0;
        d.SystemTestInitialized = true;
        d.SystemTestActive = true;

        std::cout << "[SystemTest] joint order trajectory started" << std::endl;
        std::cout << "[SystemTest] baseline=current motor position. One joint moves at a time in device order." << std::endl;
        std::cout << "[SystemTest] SAFETY: run on the real robot only when suspended or firmly supported." << std::endl;
        for (size_t i = 0; i < JOINT_NUMBER; ++i) {
            std::cout << "[SystemTestMap] index=" << i
                      << " joint=" << kDeviceOrderJointNames[i]
                      << " amplitude_rad=" << SystemTestAmplitude(i)
                      << std::endl;
        }
    }

    const RealNumber dt = static_cast<RealNumber>(d.TaskScheduler->getSpinOnceTime());
    const size_t segment_steps = std::max<size_t>(1, static_cast<size_t>(std::llround(kJointSegmentSeconds / dt)));
    const size_t hold_steps = std::max<size_t>(1, static_cast<size_t>(std::llround(kInitialHoldSeconds / dt)));

    MotorVec target = d.SystemTestBaseline;
    MotorVec zero = MotorVec::zeros();
    size_t active_joint = JOINT_NUMBER;
    RealNumber offset = 0;

    if (d.SystemTestStep >= hold_steps) {
        const size_t active_step = d.SystemTestStep - hold_steps;
        active_joint = active_step / segment_steps;
        const size_t local_step = active_step % segment_steps;

        if (active_joint < JOINT_NUMBER) {
            const RealNumber phase = static_cast<RealNumber>(local_step) / static_cast<RealNumber>(segment_steps);
            offset = SystemTestAmplitude(active_joint) * static_cast<RealNumber>(std::sin(2.0 * kPi * phase));
            target[active_joint] += offset;

            if (local_step == 0) {
                std::cout << "[SystemTest] active_index=" << active_joint
                          << " joint=" << kDeviceOrderJointNames[active_joint]
                          << " amplitude_rad=" << SystemTestAmplitude(active_joint)
                          << std::endl;
            }

            const size_t print_interval = std::max<size_t>(1, segment_steps / 4);
            if (local_step % print_interval == 0) {
                MotorVec current;
                d.TaskScheduler->template GetData<"CurrentMotorPosition">(current);
                std::cout << "[SystemTestSample] index=" << active_joint
                          << " joint=" << kDeviceOrderJointNames[active_joint]
                          << " offset_rad=" << offset
                          << " target_rad=" << target[active_joint]
                          << " current_rad=" << current[active_joint]
                          << std::endl;
            }
        } else if (d.SystemTestActive) {
            d.SystemTestActive = false;
            std::cout << "[SystemTest] finished. Holding baseline position. Press p to return to init_pose or t to rerun." << std::endl;
        }
    }

    d.TaskScheduler->template SetData<"TargetMotorPosition">(target);
    d.TaskScheduler->template SetData<"TargetMotorVelocity">(zero);
    d.TaskScheduler->template SetData<"TargetMotorTorque">(zero);

    ++d.SystemTestStep;
    d.TaskScheduler->SpinOnce();
}


void StatePolicyRunFunc(const bitbot::KernelInterface& kernel,
    bitbot::ExtraData& extra_data, UserData& d)
{
#ifdef BUILD_SIMULATION
    static size_t run_call_count = 0;
    ++run_call_count;
    if (d.SimDiagnosticsEnabled && d.MujocoData != nullptr && d.PelvisBodyId >= 0 && run_call_count % 250 == 0)
    {
        const mjData* mj_d = d.MujocoData;
        const int b = d.PelvisBodyId;
        std::cout << "[SimRoot] step=" << run_call_count
                  << " pos=" << mj_d->xpos[3 * b] << ","
                  << mj_d->xpos[3 * b + 1] << ","
                  << mj_d->xpos[3 * b + 2]
                  << " quat=" << mj_d->xquat[4 * b + 0] << ","
                  << mj_d->xquat[4 * b + 1] << ","
                  << mj_d->xquat[4 * b + 2] << ","
                  << mj_d->xquat[4 * b + 3]
                  << std::endl;
    }
#endif
    d.TaskScheduler->SpinOnce(); //运行状态(控制主状态)，进行一次调度
};

void StatePolicyDanceFunc(const bitbot::KernelInterface& kernel,
    bitbot::ExtraData& extra_data, UserData& d)
{
    d.TaskScheduler->SpinOnce(); //运行状态(跳舞主状态)，进行一次调度
}

void StateInitPoseFunc(const bitbot::KernelInterface& kernel,
    bitbot::ExtraData& extra_data, UserData& d)
{
    d.TaskScheduler->SpinOnce(); //复位状态，进行一次调度
}
