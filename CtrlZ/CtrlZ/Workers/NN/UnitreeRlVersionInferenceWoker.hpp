/**
 * @file UnitreeRlLabVelocityInferenceWorker.hpp
 * @author Zishun Zhou
 * @brief
 *
 * @date 2026-01-09
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

#include "CommonLocoInferenceWorker.hpp"
#include "NetInferenceWorker.h"
#include "Utils/ZenBuffer.hpp"
#include "Utils/StaticStringUtils.hpp"
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <numeric>

namespace z
{
    template<typename SchedulerType, CTString NetName, typename InferencePrecision, size_t INPUT_STUCK_LENGTH, size_t JOINT_NUMBER>
    class UnitreeRlVersionInferenceWorker : public CommonLocoInferenceWorker<SchedulerType, NetName, InferencePrecision, JOINT_NUMBER>
    {
    public:
        using MotorValVec = math::Vector<InferencePrecision, JOINT_NUMBER>;
        using ValVec3 = math::Vector<InferencePrecision, 3>;

    public:
        UnitreeRlVersionInferenceWorker(SchedulerType::Ptr scheduler, const nlohmann::json& Net_Config, const nlohmann::json& Motor_Config)
            :CommonLocoInferenceWorker<SchedulerType, NetName, InferencePrecision, JOINT_NUMBER>(scheduler, Net_Config, Motor_Config),
            GravityVector({ 0.0,0.0,-1.0 }),
            ang_vel_buffer(INPUT_STUCK_LENGTH),
            projected_gravity_buffer(INPUT_STUCK_LENGTH),
            velo_cmd_buffer(INPUT_STUCK_LENGTH),
            motor_pos_buffer(INPUT_STUCK_LENGTH),
            motor_vel_buffer(INPUT_STUCK_LENGTH),
            last_action_buffer(INPUT_STUCK_LENGTH)
        {
            //read cfg
            nlohmann::json InferenceCfg = Net_Config["Inference"];
            nlohmann::json NetworkCfg = Net_Config["Network"];
              nlohmann::json PreprocessCfg = Net_Config["Preprocess"];
            this->dt = scheduler->getSpinOnceTime();
              this->ImuYawCorrectionRad = PreprocessCfg.value("ImuYawCorrectionRad", static_cast<InferencePrecision>(0.0));
              this->ForceDepthValue = PreprocessCfg.value("ForceDepthValue", static_cast<InferencePrecision>(-1.0));
              this->ReverseProprioHistory = PreprocessCfg.value("ReverseProprioHistory", false);
              this->UseAlterImu = PreprocessCfg.value("UseAlterImu", false);
              this->InvertAngVelZ = PreprocessCfg.value("InvertAngVelZ", false);
              this->UseMujocoRootQvelAngVel = PreprocessCfg.value("UseMujocoRootQvelAngVel", false);
              this->UseImuRotationMatrixProjectedGravity = PreprocessCfg.value("UseImuRotationMatrixProjectedGravity", true);
              this->HeadingHoldKp = PreprocessCfg.value("HeadingHoldKp", static_cast<InferencePrecision>(0.0));
              this->HeadingHoldMaxYawRate = PreprocessCfg.value("HeadingHoldMaxYawRate", static_cast<InferencePrecision>(0.2));
              this->HeadingHoldCommandThreshold = PreprocessCfg.value("HeadingHoldCommandThreshold", static_cast<InferencePrecision>(0.02));
              this->PolicyWarmupSteps = PreprocessCfg.value("PolicyWarmupSteps", static_cast<size_t>(0));
              this->DebugPrintEnabled = PreprocessCfg.value("DebugPrint", false);
              this->HeadingHoldUseFixedTarget = PreprocessCfg.contains("HeadingHoldTargetYawRad");
              this->HeadingHoldTargetYaw = PreprocessCfg.value("HeadingHoldTargetYawRad", static_cast<InferencePrecision>(0.0));

            this->PrintSplitLine();
            std::cout << "UnitreeRlLabVelocityInferenceWorker" << std::endl;
            std::cout << "JOINT_NUMBER=" << JOINT_NUMBER << std::endl;
            std::cout << "dt=" << this->dt << std::endl;
              std::cout << "ImuYawCorrectionRad=" << this->ImuYawCorrectionRad << std::endl;
                std::cout << "ForceDepthValue=" << this->ForceDepthValue << std::endl;
                std::cout << "ReverseProprioHistory=" << this->ReverseProprioHistory << std::endl;
              std::cout << "UseAlterImu=" << this->UseAlterImu << std::endl;
              std::cout << "InvertAngVelZ=" << this->InvertAngVelZ << std::endl;
              std::cout << "UseMujocoRootQvelAngVel=" << this->UseMujocoRootQvelAngVel << std::endl;
              std::cout << "UseImuRotationMatrixProjectedGravity=" << this->UseImuRotationMatrixProjectedGravity << std::endl;
              std::cout << "HeadingHoldKp=" << this->HeadingHoldKp << std::endl;
              std::cout << "HeadingHoldMaxYawRate=" << this->HeadingHoldMaxYawRate << std::endl;
              std::cout << "HeadingHoldCommandThreshold=" << this->HeadingHoldCommandThreshold << std::endl;
              std::cout << "PolicyWarmupSteps=" << this->PolicyWarmupSteps << std::endl;
              std::cout << "HeadingHoldUseFixedTarget=" << this->HeadingHoldUseFixedTarget << std::endl;
              std::cout << "HeadingHoldTargetYaw=" << this->HeadingHoldTargetYaw << std::endl;
            this->PrintSplitLine();


            //concatenate all scales

            auto Scales_ang_vel_rep = this->Scales_ang_vel.template repeat<INPUT_STUCK_LENGTH>();
            auto Scales_project_gravity_rep = this->Scales_project_gravity.template repeat<INPUT_STUCK_LENGTH>();
            auto Scales_dof_pos_rep = this->Scales_dof_pos.template repeat<INPUT_STUCK_LENGTH>();
            auto Scales_dof_vel_rep = this->Scales_dof_vel.template repeat<INPUT_STUCK_LENGTH>();
            auto Scales_last_action_rep = this->Scales_last_action.template repeat<INPUT_STUCK_LENGTH>();
            // InstinctLab parkour config does not scale velocity_commands; only base_ang_vel uses 0.25.
            auto Scales_command3_rep = ValVec3::ones().template repeat<INPUT_STUCK_LENGTH>();


            this->InputScaleVec = math::cat(
                Scales_ang_vel_rep,
                Scales_project_gravity_rep,
                Scales_command3_rep,
                Scales_dof_pos_rep,
                Scales_dof_vel_rep,
                Scales_last_action_rep
            );

            this->TrainingJointDefaultPos = DeviceToTrain(this->JointDefaultPos);
            this->TrainingActionScale = TrainingActionScaleReference();
            this->TrainingJointClipUpper = TrainingJointClipUpperReference();
            this->TrainingJointClipLower = TrainingJointClipLowerReference();
            if (this->DebugPrintEnabled) {
                std::cout << "TrainingActionScale(reference)=" << this->TrainingActionScale << std::endl;
            }

            // Base class loads actor.onnx as the primary session. The depth encoder is loaded here.
            std::filesystem::path actor_model_path = NetworkCfg["ModelPath"].get<std::string>();
            std::filesystem::path encoder_model_path =
                NetworkCfg.contains("EncoderModelPath")
                    ? std::filesystem::path(NetworkCfg["EncoderModelPath"].get<std::string>())
                    : actor_model_path.parent_path() / "0-depth_encoder.onnx";
            if (!encoder_model_path.is_absolute()) {
                encoder_model_path = std::filesystem::absolute(encoder_model_path);
            }

#ifndef USE_OPENVINO
            EncoderSession__ = Ort::Session(GetOrtEnv(), encoder_model_path.string().c_str(), this->SessionOptions__);
            EncoderInputNodeName__ = NetworkCfg.value("EncoderInputNodeName", "input");
            EncoderOutputNodeName__ = NetworkCfg.value("EncoderOutputNodeName", GetFirstOutputName(EncoderSession__));
            this->InputNodeNames__ = { NetworkCfg.value("ActorInputNodeName", GetFirstInputName(this->Session__)) };
            this->OutputNodeNames__ = { NetworkCfg.value("ActorOutputNodeName", GetFirstOutputName(this->Session__)) };
#else
            throw(std::runtime_error("UnitreeRlVersionInferenceWorker currently supports ONNXRuntime only."));
#endif

            // Actor input: proprio history plus 128-d depth latent, matching InstinctLab export script.
            this->InputOrtTensors__.push_back(this->WarpOrtTensor(ActorInputTensor));
            this->OutputOrtTensors__.push_back(this->WarpOrtTensor(OutputTensor));

            EncoderInputOrtTensors__.push_back(this->WarpOrtTensor(DepthInputTensor));
            EncoderOutputOrtTensors__.push_back(this->WarpOrtTensor(DepthLatentTensor));
        }

        virtual ~UnitreeRlVersionInferenceWorker()
        {

        }

        void PreProcess() override
        {
            this->start_time = std::chrono::steady_clock::now();

            MotorValVec CurrentMotorVel;
            this->Scheduler->template GetData<"CurrentMotorVelocity">(CurrentMotorVel);
            DebugCurrentMotorVelDevice = CurrentMotorVel;
            CurrentMotorVel = DeviceToTrain(CurrentMotorVel);

            MotorValVec CurrentMotorPos;
            this->Scheduler->template GetData<"CurrentMotorPosition">(CurrentMotorPos);
            DebugCurrentMotorPosDevice = CurrentMotorPos;
            CurrentMotorPos = DeviceToTrain(CurrentMotorPos);
            DebugCurrentMotorPosTrain = CurrentMotorPos;
            CurrentMotorPos -= this->TrainingJointDefaultPos;

            MotorValVec LastAction;
            this->Scheduler->template GetData<concat(NetName, "NetLastAction")>(LastAction);

            ValVec3 UserCmd3;
            this->Scheduler->template GetData<concat(NetName, "NetUserCommand3")>(UserCmd3);

            // ValVec3 LinVel;
            // this->Scheduler->template GetData<"LinearVelocityValue">(LinVel);

            ValVec3 AngVel;
            if (this->UseAlterImu) {
                this->Scheduler->template GetData<"AlterAngleVelocityValue">(AngVel);
            } else {
                this->Scheduler->template GetData<"AngleVelocityValue">(AngVel);
            }
            if (this->UseMujocoRootQvelAngVel) {
                this->Scheduler->template GetData<"MujocoRootAngVelValue">(AngVel);
            }
            if (this->InvertAngVelZ) {
                AngVel[2] = -AngVel[2];
            }

            ValVec3 Ang;
            if (this->UseAlterImu) {
                this->Scheduler->template GetData<"AlterAngleValue">(Ang);
            } else {
                this->Scheduler->template GetData<"AngleValue">(Ang);
            }

            if (this->HeadingHoldKp != static_cast<InferencePrecision>(0.0)) {
                const InferencePrecision planar_cmd_norm = std::sqrt(UserCmd3[0] * UserCmd3[0] + UserCmd3[1] * UserCmd3[1]);
                const bool has_translation_cmd = planar_cmd_norm > this->HeadingHoldCommandThreshold;
                const bool has_manual_yaw_cmd = std::abs(UserCmd3[2]) > this->HeadingHoldCommandThreshold;

                if (!has_translation_cmd || has_manual_yaw_cmd) {
                    if (!this->HeadingHoldUseFixedTarget) {
                        this->HeadingHoldTargetYaw = Ang[2];
                    }
                    this->HeadingHoldInitialized = false;
                } else {
                if (!this->HeadingHoldInitialized) {
                    if (!this->HeadingHoldUseFixedTarget) {
                        this->HeadingHoldTargetYaw = Ang[2];
                    }
                    this->HeadingHoldInitialized = true;
                }
                const InferencePrecision yaw_error = WrapToPi(this->HeadingHoldTargetYaw - Ang[2]);
                UserCmd3[2] += this->HeadingHoldKp * yaw_error;
                if (UserCmd3[2] > this->HeadingHoldMaxYawRate) {
                    UserCmd3[2] = this->HeadingHoldMaxYawRate;
                } else if (UserCmd3[2] < -this->HeadingHoldMaxYawRate) {
                    UserCmd3[2] = -this->HeadingHoldMaxYawRate;
                }
                }
            }
            DebugUserCmd3 = UserCmd3;

            ValVec3 ProjectedGravity = ComputeProjectedGravity(Ang, this->GravityVector);
            if (this->UseImuRotationMatrixProjectedGravity)
            {
                z::math::Vector<InferencePrecision, 9> RotMat;
                if (this->UseAlterImu) {
                    this->Scheduler->template GetData<"AlterRotationMatrixValue">(RotMat);
                } else {
                    this->Scheduler->template GetData<"RotationMatrixValue">(RotMat);
                }
                ProjectedGravity = ValVec3({-RotMat[6], -RotMat[7], -RotMat[8]});
            }
              AngVel = RotateYawCorrection(AngVel);
              ProjectedGravity = RotateYawCorrection(ProjectedGravity);
            this->Scheduler->template SetData<concat(NetName, "NetProjectedGravity")>(ProjectedGravity);


            if (!HistoryInitialized)
            {
                for (size_t i = 0; i < INPUT_STUCK_LENGTH; ++i)
                {
                    this->ang_vel_buffer.push(AngVel);
                    this->projected_gravity_buffer.push(ProjectedGravity);
                    this->velo_cmd_buffer.push(UserCmd3);
                    this->motor_pos_buffer.push(CurrentMotorPos);
                    this->motor_vel_buffer.push(CurrentMotorVel);
                    this->last_action_buffer.push(LastAction);
                }
                HistoryInitialized = true;
            }
            else
            {
                this->ang_vel_buffer.push(AngVel);
                this->projected_gravity_buffer.push(ProjectedGravity);
                this->velo_cmd_buffer.push(UserCmd3);
                this->motor_pos_buffer.push(CurrentMotorPos);
                this->motor_vel_buffer.push(CurrentMotorVel);
                this->last_action_buffer.push(LastAction);
            }

            // std::cout << "Input Buffer State:" << std::endl;
            // std::cout << "AngVel Buffer: " << this->ang_vel_buffer;

            auto ang_vel_stacked = this->stackBuffer(this->ang_vel_buffer);
            auto projected_gravity_stacked = this->stackBuffer(this->projected_gravity_buffer);
            auto velo_cmd_stacked = this->stackBuffer(this->velo_cmd_buffer);
            auto motor_pos_stacked = this->stackBuffer(this->motor_pos_buffer);
            auto motor_vel_stacked = this->stackBuffer(this->motor_vel_buffer);
            auto last_action_stacked = this->stackBuffer(this->last_action_buffer);

            auto ProprioInputVec = math::cat(
                ang_vel_stacked,
                projected_gravity_stacked,
                velo_cmd_stacked,
                motor_pos_stacked,
                motor_vel_stacked,
                last_action_stacked
            ) * this->InputScaleVec;

            this->ProprioInputTensor.Array() = z::math::clamp(ProprioInputVec, -this->ClipObservation, this->ClipObservation);

            DepthImageVec DepthObs;
            this->Scheduler->template GetData<"ParkourDepthImage">(DepthObs);
            for (size_t i = 0; i < DEPTH_TENSOR_LENGTH; ++i) {
                  DepthInputTensor[i] = this->ForceDepthValue >= static_cast<InferencePrecision>(0.0)
                      ? this->ForceDepthValue
                      : static_cast<InferencePrecision>(DepthObs[i]);
            }
        }

        void InferenceOnce() override
        {
#ifndef USE_OPENVINO
            const char* encoder_input_names[] = { EncoderInputNodeName__.c_str() };
            const char* encoder_output_names[] = { EncoderOutputNodeName__.c_str() };
            EncoderSession__.Run(Ort::RunOptions{ nullptr },
                                 encoder_input_names,
                                 EncoderInputOrtTensors__.data(),
                                 EncoderInputOrtTensors__.size(),
                                 encoder_output_names,
                                 EncoderOutputOrtTensors__.data(),
                                 EncoderOutputOrtTensors__.size());

            for (size_t i = 0; i < PROPRIO_TENSOR_LENGTH; ++i) {
                ActorInputTensor[i] = ProprioInputTensor[i];
            }
            for (size_t i = 0; i < DEPTH_LATENT_LENGTH; ++i) {
                ActorInputTensor[PROPRIO_TENSOR_LENGTH + i] = DepthLatentTensor[i];
            }

            this->Session__.Run(Ort::RunOptions{ nullptr }, this->IoBinding__);
#endif
        }

        template<typename Scalar, size_t N>
        z::math::Vector<Scalar, N* INPUT_STUCK_LENGTH> stackBuffer(z::RingBuffer<z::math::Vector<Scalar, N>> buffer)
        {
            z::math::Vector<Scalar, N* INPUT_STUCK_LENGTH> result;
            for (size_t i = 0; i < INPUT_STUCK_LENGTH; i++)
            {
                const size_t history_idx = this->ReverseProprioHistory ? (INPUT_STUCK_LENGTH - 1 - i) : i;
                std::copy(buffer[history_idx].begin(), buffer[history_idx].end(), result.begin() + i * N);
            }
            return result;
        }

        void PostProcess() override
        {
            auto LastAction = this->OutputTensor.toVector();
            MotorValVec ClipedLastAction;
            MotorValVec DeviceOrderAction;

            if (this->PolicyStepCount < this->PolicyWarmupSteps)
            {
                ClipedLastAction = MotorValVec::zeros();
                this->Scheduler->template SetData<concat(NetName, "NetLastAction")>(ClipedLastAction);
                DeviceOrderAction = TrainToDevice(this->TrainingJointDefaultPos);
                ++this->PolicyStepCount;
            }
            else
            {
                ClipedLastAction = z::math::clamp(LastAction, -this->ClipAction, this->ClipAction);
                this->Scheduler->template SetData<concat(NetName, "NetLastAction")>(ClipedLastAction);

                auto ScaledAction = ClipedLastAction * this->TrainingActionScale + this->TrainingJointDefaultPos;
                auto clipedAction = z::math::clamp(ScaledAction, this->TrainingJointClipLower, this->TrainingJointClipUpper);
                DeviceOrderAction = TrainToDevice(clipedAction);
            }
            this->Scheduler->template SetData<concat(NetName, "Action")>(DeviceOrderAction);

            if (this->DebugPrintEnabled && DebugPrintCount < 20 && DebugTick++ % 50 == 0)
            {
                ValVec3 ProjectedGravity;
                this->Scheduler->template GetData<concat(NetName, "NetProjectedGravity")>(ProjectedGravity);
                ValVec3 AngVel;
                this->Scheduler->template GetData<"AngleVelocityValue">(AngVel);
                ValVec3 Ang;
                this->Scheduler->template GetData<"AngleValue">(Ang);

                std::cout << "[VersionPolicyDebug] tick=" << DebugTick
                          << " angle=" << Ang
                          << " projected_gravity=" << ProjectedGravity
                          << " ang_vel=" << AngVel
                          << " user_cmd=" << DebugUserCmd3;
                PrintStats(" proprio", ProprioInputTensor);
                PrintStats(" depth", DepthInputTensor);
                PrintStats(" raw_action", OutputTensor);
                PrintStats(" target_device", DeviceOrderAction);
                std::cout << " pos_device_head=[";
                for (size_t i = 0; i < std::min<size_t>(JOINT_NUMBER, 12); ++i)
                {
                    std::cout << DebugCurrentMotorPosDevice[i] << (i + 1 < std::min<size_t>(JOINT_NUMBER, 12) ? "," : "");
                }
                std::cout << "] pos_train_head=[";
                for (size_t i = 0; i < std::min<size_t>(JOINT_NUMBER, 12); ++i)
                {
                    std::cout << DebugCurrentMotorPosTrain[i] << (i + 1 < std::min<size_t>(JOINT_NUMBER, 12) ? "," : "");
                }
                std::cout << "]";
                std::cout << " target_head=[";
                for (size_t i = 0; i < std::min<size_t>(JOINT_NUMBER, 12); ++i)
                {
                    std::cout << DeviceOrderAction[i] << (i + 1 < std::min<size_t>(JOINT_NUMBER, 12) ? "," : "");
                }
                std::cout << "]" << std::endl;
                ++DebugPrintCount;
            }

            this->end_time = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(this->end_time - this->start_time);
            InferencePrecision inference_time = static_cast<InferencePrecision>(duration.count()) / 1000.0;
            this->Scheduler->template SetData<concat(NetName, "InferenceTime")>(inference_time);
        }

    private:
        //base ang vel; proj grav;velo_cmd; dof pos; dof vel; last action
        static constexpr size_t INPUT_TENSOR_LENGTH_UNIT = 3 + 3 + 3 + JOINT_NUMBER + JOINT_NUMBER + JOINT_NUMBER;
        static constexpr size_t PROPRIO_TENSOR_LENGTH = INPUT_TENSOR_LENGTH_UNIT * INPUT_STUCK_LENGTH;
        static constexpr size_t DEPTH_HISTORY_LENGTH = 8;
        static constexpr size_t DEPTH_HEIGHT = 18;
        static constexpr size_t DEPTH_WIDTH = 32;
        static constexpr size_t DEPTH_TENSOR_LENGTH = DEPTH_HISTORY_LENGTH * DEPTH_HEIGHT * DEPTH_WIDTH;
        static constexpr size_t DEPTH_LATENT_LENGTH = 128;
        static constexpr size_t ACTOR_INPUT_TENSOR_LENGTH = PROPRIO_TENSOR_LENGTH + DEPTH_LATENT_LENGTH;
        //joint number
        static constexpr size_t OUTPUT_TENSOR_LENGTH = JOINT_NUMBER;
        using DepthImageVec = math::Vector<InferencePrecision, DEPTH_TENSOR_LENGTH>;

        // Must match mujoco_sim2sim/sim2sim/constants.py TRAINING_JOINT_ORDER.
        // This maps the policy/training joint index to bitbot_mujoco.xml device order.
        static constexpr std::array<size_t, JOINT_NUMBER> TrainToDeviceIndex = {
            15, 22, 14, 16, 23, 13, 17, 24, 12, 18, 25,
            0, 6, 19, 26, 1, 7, 20, 27, 2, 8, 21, 28,
            3, 9, 4, 10, 5, 11
        };

        static MotorValVec TrainingActionScaleReference()
        {
            return MotorValVec({
                0.4385773139, 0.4385773139, 0.4385773139, 0.4385773139, 0.4385773139,
                0.4385773139, 0.4385773139, 0.4385773139, 0.5475464652, 0.4385773139,
                0.4385773139, 0.5475464652, 0.5475464652, 0.4385773139, 0.4385773139,
                0.3506614664, 0.3506614664, 0.07450087033, 0.07450087033, 0.5475464652,
                0.5475464652, 0.07450087033, 0.07450087033, 0.3506614664, 0.3506614664,
                0.4385773139, 0.4385773139, 0.4385773139, 0.4385773139
            });
        }

        static MotorValVec TrainingJointClipUpperReference()
        {
            return MotorValVec({
                2.6704, 2.6704, 0.52, 2.2515, 1.5882, 0.52, 2.618, 2.618, 2.618,
                2.0944, 2.0944, 2.8798, 2.8798, 1.97222, 1.97222, 2.9671, 0.5236,
                1.61443, 1.61443, 2.7576, 2.7576, 1.61443, 1.61443, 2.8798, 2.8798,
                0.5236, 0.5236, 0.2618, 0.2618
            });
        }

        static MotorValVec TrainingJointClipLowerReference()
        {
            return MotorValVec({
                -3.0892, -3.0892, -0.52, -1.5882, -2.2515, -0.52, -2.618, -2.618,
                -2.618, -1.0472, -1.0472, -2.5307, -2.5307, -1.97222, -1.97222,
                -0.5236, -2.9671, -1.61443, -1.61443, -2.7576, -2.7576, -1.61443,
                -1.61443, -0.087267, -0.087267, -0.87267, -0.87267, -0.2618, -0.2618
            });
        }

        static MotorValVec DeviceToTrain(const MotorValVec& device_order)
        {
            MotorValVec train_order;
            for (size_t train_idx = 0; train_idx < JOINT_NUMBER; ++train_idx)
            {
                train_order[train_idx] = device_order[TrainToDeviceIndex[train_idx]];
            }
            return train_order;
        }

        static MotorValVec TrainToDevice(const MotorValVec& train_order)
        {
            MotorValVec device_order;
            for (size_t train_idx = 0; train_idx < JOINT_NUMBER; ++train_idx)
            {
                device_order[TrainToDeviceIndex[train_idx]] = train_order[train_idx];
            }
            return device_order;
        }

          ValVec3 RotateYawCorrection(const ValVec3& value) const
          {
              const InferencePrecision c = static_cast<InferencePrecision>(std::cos(this->ImuYawCorrectionRad));
              const InferencePrecision s = static_cast<InferencePrecision>(std::sin(this->ImuYawCorrectionRad));
              return ValVec3({
                  c * value[0] - s * value[1],
                  s * value[0] + c * value[1],
                  value[2]
              });
          }

          static InferencePrecision WrapToPi(InferencePrecision angle)
          {
              return static_cast<InferencePrecision>(std::atan2(std::sin(angle), std::cos(angle)));
          }

        template<typename Container>
        static void PrintStats(const char* name, Container& values)
        {
            InferencePrecision min_val = std::numeric_limits<InferencePrecision>::max();
            InferencePrecision max_val = std::numeric_limits<InferencePrecision>::lowest();
            InferencePrecision sum_val = 0;
            const auto* data = values.data();
            for (size_t i = 0; i < values.size(); ++i)
            {
                const InferencePrecision v = static_cast<InferencePrecision>(data[i]);
                min_val = std::min(min_val, v);
                max_val = std::max(max_val, v);
                sum_val += v;
            }
            std::cout << name << "_min=" << min_val
                      << name << "_max=" << max_val
                      << name << "_mean=" << sum_val / static_cast<InferencePrecision>(values.size());
        }

        //input tensor
        z::math::Tensor<InferencePrecision, 1, PROPRIO_TENSOR_LENGTH> ProprioInputTensor;
        z::math::Tensor<InferencePrecision, 1, DEPTH_HISTORY_LENGTH, DEPTH_HEIGHT, DEPTH_WIDTH> DepthInputTensor;
        z::math::Tensor<InferencePrecision, 1, ACTOR_INPUT_TENSOR_LENGTH> ActorInputTensor;
        z::math::Vector<InferencePrecision, PROPRIO_TENSOR_LENGTH> InputScaleVec;

        z::RingBuffer<ValVec3> ang_vel_buffer;
        z::RingBuffer<ValVec3> projected_gravity_buffer;
        z::RingBuffer<ValVec3> velo_cmd_buffer;
        z::RingBuffer<MotorValVec> motor_pos_buffer;
        z::RingBuffer<MotorValVec> motor_vel_buffer;
        z::RingBuffer<MotorValVec> last_action_buffer;

        //output tensor
        z::math::Tensor<InferencePrecision, 1, DEPTH_LATENT_LENGTH> DepthLatentTensor;
        z::math::Tensor<InferencePrecision, 1, OUTPUT_TENSOR_LENGTH> OutputTensor;
        MotorValVec TrainingJointDefaultPos;
        MotorValVec TrainingActionScale;
        MotorValVec TrainingJointClipUpper;
        MotorValVec TrainingJointClipLower;
        ValVec3 DebugUserCmd3;
        MotorValVec DebugCurrentMotorPosDevice;
        MotorValVec DebugCurrentMotorPosTrain;
        MotorValVec DebugCurrentMotorVelDevice;
          InferencePrecision ImuYawCorrectionRad = 0;
          InferencePrecision ForceDepthValue = -1;
          bool HistoryInitialized = false;
          bool ReverseProprioHistory = false;
          bool UseAlterImu = false;
          bool InvertAngVelZ = false;
          bool UseMujocoRootQvelAngVel = false;
          bool UseImuRotationMatrixProjectedGravity = true;
          InferencePrecision HeadingHoldKp = 0;
          InferencePrecision HeadingHoldMaxYawRate = static_cast<InferencePrecision>(0.2);
          InferencePrecision HeadingHoldCommandThreshold = static_cast<InferencePrecision>(0.02);
          size_t PolicyWarmupSteps = 0;
          size_t PolicyStepCount = 0;
          bool HeadingHoldUseFixedTarget = false;
          bool HeadingHoldInitialized = false;
          InferencePrecision HeadingHoldTargetYaw = 0;
        size_t DebugTick = 0;
        size_t DebugPrintCount = 0;
        bool DebugPrintEnabled = false;

#ifndef USE_OPENVINO
        Ort::Session EncoderSession__ = Ort::Session(nullptr);
        std::string EncoderInputNodeName__;
        std::string EncoderOutputNodeName__;
        std::vector<Ort::Value> EncoderInputOrtTensors__;
        std::vector<Ort::Value> EncoderOutputOrtTensors__;

        std::string GetFirstOutputName(Ort::Session& session)
        {
            Ort::AllocatorWithDefaultOptions allocator;
            auto output_name = session.GetOutputNameAllocated(0, allocator);
            return std::string(output_name.get());
        }

        std::string GetFirstInputName(Ort::Session& session)
        {
            Ort::AllocatorWithDefaultOptions allocator;
            auto input_name = session.GetInputNameAllocated(0, allocator);
            return std::string(input_name.get());
        }
#endif

        const ValVec3 GravityVector;

        InferencePrecision dt;

        //compute time
        std::chrono::steady_clock::time_point start_time;
        std::chrono::steady_clock::time_point end_time;
    };
};
