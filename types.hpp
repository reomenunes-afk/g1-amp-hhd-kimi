/**
 * @file types.hpp
 * @author Zishun Zhou
 * @brief 类型定义
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once
#include <array>

#include "Schedulers/AbstractScheduler.hpp"

#include "Utils/StaticStringUtils.hpp"
#include "Utils/MathTypes.hpp"

#include "Workers/AbstractWorker.hpp"
#include "Workers/AsyncLoggerWorker.hpp"
#include "Workers/ImuProcessWorker.hpp"
#include "Workers/MotorResetPositionWorker.hpp"
#include "Workers/NetCmdWorker.hpp"
#include "Workers/ActionManagementWorker.hpp"
#include "Workers/NN/BeyondMimicWorker.hpp"
#include "Workers/NN/UnitreeRlLabVelocityInferenceWorker.hpp"
#include "Workers/NN/UnitreeRlVersionInferenceWoker.hpp"
#include "Workers/MITMotorControlWorker.hpp"

#ifdef BUILD_SIMULATION
#include "bitbot_mujoco/device/mujoco_imu.h"
#include "bitbot_mujoco/device/mujoco_joint.h"
#include "mujoco_camera.h"
using DeviceImu = bitbot::MujocoImu;
using DeviceJoint = bitbot::MujocoJoint;
using DeviceCamera = bitbot::MujocoCamera;
using DepthFrameType = bitbot::MujocoDepthFrame;
#else
#include "Bitbot_Unitree/include/device/unitree_imu.h"
#include "Bitbot_Unitree/include/device/unitree_joint.h"
#include "Bitbot_Unitree/include/device/unitree_gamepad.h"
#include "Bitbot_Unitree/include/device/unitree_camera.h"  // ← 添加这一行
using DeviceImu = bitbot::UnitreeImu;
  using DeviceJoint = bitbot::UnitreeJoint;
  using DeviceCamera = bitbot::UnitreeCamera;
using DepthFrameType = rs2::depth_frame;
#endif



/************ basic definintion***********/
using RealNumber = float;
constexpr size_t JOINT_NUMBER = 29;
using Vec3 = z::math::Vector<RealNumber, 3>;
using Vec9 = z::math::Vector<RealNumber, 9>;
using MotorVec = z::math::Vector<RealNumber, JOINT_NUMBER>;
//constexpr size_t DANCE_TRAJECTORY_LENGTH = 1749; //NOTE: remember to change this when changing dancing trajectories dance 102
constexpr size_t DANCE_TRAJECTORY_LENGTH = 1942; //NOTE: remember to change this when changing dancing trajectories gannam style
//constexpr size_t DANCE_TRAJECTORY_LENGTH = 2897; //NOTE: remember to change this when changing dancing trajectories kuailechongbai

#ifdef BUILD_SIMULATION
// The regenerated MuJoCo model declares joints in the same order as the
// MotorVec/device-bus ordering (left leg, right leg, waist, left arm, right arm).
constexpr std::array<size_t, JOINT_NUMBER> JOINT_ID_MAP = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28
};
constexpr size_t IMU_ID_MAP = 29;
constexpr size_t ALTER_IMU_ID_MAP = 30;
constexpr size_t CAMERA_ID_MAP = 35;
#else
// Unitree SDK LowState/LowCmd motor indices and bitbot_unitree.xml device ids
// both use the same 29-DoF device order:
// left leg, right leg, waist, left arm, right arm.
constexpr std::array<size_t, JOINT_NUMBER> JOINT_ID_MAP = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28
};
constexpr size_t IMU_ID_MAP = 30; // Unitree IMU ID
constexpr size_t ALTER_IMU_ID_MAP = 31; //unitree alter IMU ID
constexpr size_t CAMERA_ID_MAP = 35; // Unitree Camera ID
#endif

/********** IMU Data Pair******************/
constexpr z::CTSPair<"AccelerationRaw", Vec3> ImuAccRawPair;
constexpr z::CTSPair<"AngleVelocityRaw", Vec3> ImuGyroRawPair;
constexpr z::CTSPair<"AngleRaw", Vec3> ImuMagRawPair;

constexpr z::CTSPair<"AccelerationValue", Vec3> ImuAccFilteredPair;
constexpr z::CTSPair<"AngleValue", Vec3> ImuMagFilteredPair;
constexpr z::CTSPair<"AngleVelocityValue", Vec3> ImuGyroFilteredPair;
constexpr z::CTSPair<"AlterAngleValue", Vec3> ImuAlterAngleFilteredPair;
constexpr z::CTSPair<"AlterAngleVelocityValue", Vec3> ImuAlterGyroFilteredPair;
constexpr z::CTSPair<"RotationMatrixValue", Vec9> ImuRotMatPair;
constexpr z::CTSPair<"AlterRotationMatrixValue", Vec9> ImuAlterRotMatPair;
constexpr z::CTSPair<"MujocoRootAngVelValue", Vec3> MujocoRootAngVelPair;

/********** Camera Data Pair******************/
constexpr z::CTSPair<"DepthCenterDistance", RealNumber> DepthCenterDistPair;
constexpr z::CTSPair<"DepthMinDistance", RealNumber> DepthMinDistPair;
constexpr z::CTSPair<"DepthMaxDistance", RealNumber> DepthMaxDistPair;
constexpr size_t PARKOUR_DEPTH_HISTORY = 8;
constexpr size_t PARKOUR_DEPTH_HEIGHT = 18;
constexpr size_t PARKOUR_DEPTH_WIDTH = 32;
using ParkourDepthImageVec = z::math::Vector<RealNumber, PARKOUR_DEPTH_HISTORY * PARKOUR_DEPTH_HEIGHT * PARKOUR_DEPTH_WIDTH>;
constexpr z::CTSPair<"ParkourDepthImage", ParkourDepthImageVec> ParkourDepthImagePair;


/********** Motor control Pair ************/
constexpr z::CTSPair<"TargetMotorPosition", MotorVec> TargetMotorPosPair;
constexpr z::CTSPair<"TargetMotorVelocity", MotorVec> TargetMotorVelPair;
constexpr z::CTSPair<"TargetMotorTorque", MotorVec> TargetMotorTorquePair;
constexpr z::CTSPair<"CurrentMotorPosition", MotorVec> CurrentMotorPosPair;
constexpr z::CTSPair<"CurrentMotorVelocity", MotorVec> CurrentMotorVelPair;
constexpr z::CTSPair<"CurrentMotorTorque", MotorVec> CurrentMotorTorquePair;
constexpr z::CTSPair<"TargetMotorStiffness", MotorVec> TargetMotorStiffnessPair;
constexpr z::CTSPair<"TargetMotorDamping", MotorVec> TargetMotorDampingPair;

/********* NN pair ********************/
constexpr z::CTString Net1Name = "DanceNet1";
constexpr z::CTSPair<z::concat(Net1Name, "NetLastAction"), MotorVec> NetLastActionPair;
constexpr z::CTSPair<z::concat(Net1Name, "Action"), MotorVec> DanceNet1OutPair;
constexpr z::CTSPair<z::concat(Net1Name, "InferenceTime"), RealNumber> InferenceTimePair;
constexpr z::CTSPair<z::concat(Net1Name, "RefTraj"), MotorVec> Net1RefTrajPair;
constexpr z::CTSPair<z::concat(Net1Name, "RefVel"), MotorVec> Net1RefVelPair;

/********* NN walk pair ********************/
constexpr z::CTString WalkNetName = "WalkNet1";
constexpr z::CTSPair<z::concat(WalkNetName, "NetLastAction"), MotorVec> WalkNetLastActionPair;
constexpr z::CTSPair<z::concat(WalkNetName, "Action"), MotorVec> WalkNet1OutPair;
constexpr z::CTSPair<z::concat(WalkNetName, "InferenceTime"), RealNumber> WalkInferenceTimePair;
constexpr z::CTSPair<z::concat(WalkNetName, "NetProjectedGravity"), Vec3> WalkNetProjectedGravityPair;
constexpr z::CTSPair<z::concat(WalkNetName, "NetUserCommand3"), Vec3> WalkNetUserCommand3Pair;

// define scheduler
using SchedulerType = z::AbstractScheduler<ImuAccRawPair, ImuGyroRawPair, ImuMagRawPair,
    ImuAccFilteredPair, ImuGyroFilteredPair, ImuMagFilteredPair,
    ImuRotMatPair, ImuAlterRotMatPair, MujocoRootAngVelPair,
    TargetMotorPosPair, TargetMotorVelPair, CurrentMotorPosPair, CurrentMotorVelPair, CurrentMotorTorquePair,
    TargetMotorTorquePair, TargetMotorDampingPair, TargetMotorStiffnessPair,
    NetLastActionPair, InferenceTimePair, DanceNet1OutPair, Net1RefTrajPair, Net1RefVelPair, ImuAlterAngleFilteredPair, ImuAlterGyroFilteredPair,
    WalkNetLastActionPair, WalkNet1OutPair, WalkInferenceTimePair, WalkNetProjectedGravityPair, WalkNetUserCommand3Pair,
    DepthCenterDistPair, DepthMinDistPair, DepthMaxDistPair, ParkourDepthImagePair>;


//define workers
using MotorResetWorkerType = z::MotorResetPositionWorker<SchedulerType, RealNumber, JOINT_NUMBER>;
using ImuWorkerType = z::ImuProcessWorker<SchedulerType, DeviceImu*, RealNumber>;
using AlterImuWorkerType = z::SimpleCallbackWorker<SchedulerType>;
using MotorWorkerType = z::MITMotorControlWorker<SchedulerType, DeviceJoint*, RealNumber, JOINT_NUMBER>;
using LoggerWorkerType = z::AsyncLoggerWorker<SchedulerType, RealNumber, ImuAccRawPair, ImuGyroRawPair, ImuMagRawPair,
    ImuAccFilteredPair, ImuGyroFilteredPair, ImuMagFilteredPair,
    TargetMotorPosPair, TargetMotorVelPair, CurrentMotorPosPair, CurrentMotorVelPair, CurrentMotorTorquePair,
    TargetMotorTorquePair, TargetMotorDampingPair, TargetMotorStiffnessPair,
    NetLastActionPair, InferenceTimePair, DanceNet1OutPair, Net1RefTrajPair, Net1RefVelPair, ImuAlterAngleFilteredPair,
    WalkNetLastActionPair, WalkNet1OutPair, WalkInferenceTimePair, WalkNetProjectedGravityPair, WalkNetUserCommand3Pair, DepthCenterDistPair, DepthMinDistPair, DepthMaxDistPair>;

using CmdWorkerType = z::NetCmdWorker<SchedulerType, RealNumber, WalkNetUserCommand3Pair>;
using ActionManagementWorkerType = z::ActionAndMotorPropertiesManagementWorker<SchedulerType, RealNumber, DanceNet1OutPair, WalkNet1OutPair>;

/******define actor net************/
using BeyondMimicUnitreeInferWorkerType = z::BeyondMimicUnitreeInferenceWorker<SchedulerType, Net1Name, RealNumber, JOINT_NUMBER, DANCE_TRAJECTORY_LENGTH>;
using UnitreeRlLabVelocityInferWorkerType = z::UnitreeRlVersionInferenceWorker<SchedulerType, WalkNetName, RealNumber, 8, JOINT_NUMBER>; // parkour version policy, stack 8 frames of proprioception history
