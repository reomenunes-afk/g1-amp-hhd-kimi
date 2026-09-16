/**
 * @file user_func.h
 * @author Zishun Zhou
 * @brief
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#ifdef BUILD_SIMULATION
#include "bitbot_mujoco/kernel/mujoco_kernel.hpp"
#else
#include "Bitbot_Unitree/include/kernel/unitree_kernel.hpp"
#endif

#include "types.hpp"


enum class States : bitbot::StateId
{
    StateWaiting = 1001,
    StateInitPose,
    StatePolicyRun,
    StatePolicyDance,
    StateSystemTest,
};

struct UserData
{
    SchedulerType::Ptr TaskScheduler;
    ImuWorkerType* ImuWorker;
    AlterImuWorkerType* AlterImuWorker;
    MotorWorkerType* MotorWorker;
    LoggerWorkerType* Logger;
    BeyondMimicUnitreeInferWorkerType* DanceNetInferWorker;
    HumanoidGymAmpInferWorkerType* WalkNetInferWorker;
    MotorResetWorkerType* MotorResetWorker;
    ActionManagementWorkerType* ActionManagementWorker;
    CmdWorkerType* WalkCmdWorker;
    AlterImuWorkerType* CameraWorker;
    //NOTE: you don't need to delete these workers, they will be deleted by the scheduler automaticlly when the scheduler is destroyed

    std::array<DeviceJoint*, JOINT_NUMBER> JointsPtr;
    DeviceImu* ImuPtr;
    DeviceImu* ImuAlterPtr;
    DeviceCamera* CameraPtr;
      bool AutoDebugEnabled = false;
      bool AutoDebugBusPrimed = false;
      bool AutoDebugStarted = false;
      bool AutoDebugUseReset = true;
      bool AutoDebugResetActive = false;
      bool AutoDebugPolicyActive = false;
    bool AutoDebugPolicySwitchDone = false;
    size_t AutoDebugWaitSteps = 0;
    size_t AutoDebugResetSteps = 0;
      size_t AutoDebugDelaySteps = 0;
    size_t AutoDebugDepthWarmupFrames = 0;
    size_t AutoDebugCameraFrames = 0;
    size_t AutoDebugDepthHistorySize = 0;
    bool AutoDebugDepthReady = false;
    RealNumber AutoDebugDepthMin = 0;
    RealNumber AutoDebugDepthMax = 0;
    RealNumber AutoDebugDepthMean = 0;
    bool SimDiagnosticsEnabled = false;
    bool SystemTestActive = false;
    bool SystemTestInitialized = false;
    size_t SystemTestStep = 0;
    MotorVec SystemTestBaseline = MotorVec::zeros();
    MotorVec SystemTestPeakAbsDelta = MotorVec::zeros();

#ifdef BUILD_SIMULATION
    // Optional MuJoCo diagnostics (valid only in simulation builds).
    mjData* MujocoData = nullptr;
    const mjModel* MujocoModel = nullptr;
    int PelvisBodyId = -1;
#endif
};

#ifdef BUILD_SIMULATION
using KernelType = bitbot::MujocoKernel<UserData>;
using KernelBus = bitbot::MujocoBus;
#else
using KernelType = bitbot::UnitreeKernel<UserData>;
using KernelBus = bitbot::UnitreeBus;
#endif


void ConfigFunc(const KernelBus& bus, UserData& d);
void FinishFunc(UserData& d);

void StateWaitingFunc(const bitbot::KernelInterface& kernel, bitbot::ExtraData& extra_data, UserData& user_data);
void StateInitPoseFunc(const bitbot::KernelInterface& kernel, bitbot::ExtraData& extra_data, UserData& user_data);
void StatePolicyRunFunc(const bitbot::KernelInterface& kernel, bitbot::ExtraData& extra_data, UserData& user_data);
void StatePolicyDanceFunc(const bitbot::KernelInterface& kernel, bitbot::ExtraData& extra_data, UserData& user_data);
void StateSystemTestFunc(const bitbot::KernelInterface& kernel, bitbot::ExtraData& extra_data, UserData& user_data);
