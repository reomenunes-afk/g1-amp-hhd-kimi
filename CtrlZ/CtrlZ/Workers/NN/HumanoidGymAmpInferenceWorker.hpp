/**
 * @file HumanoidGymAmpInferenceWorker.hpp
 * @brief HumanoidGym AMP (v153 actor projected-gravity) inference worker for Unitree G1 29DoF.
 *
 * Observation contract (must match training, verified against humanoid-gym worktree
 * branch AMP-v153-actor-projected-gravity, commit 6cdf195f):
 *   INPUT_STUCK_LENGTH frames x 98 dims, history ordered oldest -> newest.
 *   Single frame layout:
 *     [0:1]   sin(2*pi*phase)
 *     [1:2]   cos(2*pi*phase)          phase = step * PolicyDt / Cycle_time, phase = 0 at policy (re)start
 *     [2:5]   command [vx, vy, yaw_rate] * [2, 2, 1]
 *     [5:34]  (q - default_q) * 1
 *     [34:63] dq * 0.05
 *     [63:92] previous processed action (clipped, dimensionless)
 *     [92:95] base (pelvis) angular velocity, body frame, * 1
 *     [95:98] base (pelvis) projected gravity (unit vector), * 1
 *   The full stacked input is clamped to [-ClipObservations, ClipObservations] (18 in training).
 *
 * The IMU of this framework (both UnitreeImu on the real robot and MujocoImu site "imu")
 * is mounted in the torso, while the training base body is the pelvis. Waist compensation
 * (pelvis -> waist_yaw(z) -> waist_roll(x) -> waist_pitch(y) -> torso_link, all joint
 * origins have zero rpy in the training URDF g1_29dof_rev_1_0.urdf) maps torso-frame
 * IMU data into the pelvis frame:
 *   v_pelvis = Rz(q_yaw) * Rx(q_roll) * Ry(q_pitch) * v_torso
 *   omega_pelvis = R_w * (omega_torso - omega_rel),
 *   omega_rel(torso) = Ry(-q_pitch)*Rx(-q_roll)*[0,0,dq_yaw] + Ry(-q_pitch)*[dq_roll,0,0] + [0,dq_pitch,0]
 * (both formulas verified numerically against MuJoCo FK, err < 1e-14)
 * This uses only joint encoders + IMU, so the code path is identical on sim and real robot.
 *
 * Joint order: this worker assumes device order == training DOF order
 * (left leg 6, right leg 6, waist 3, left arm 7, right arm 7), which holds for this repo
 * (settings/bitbot_*.xml device ids 0..28) and for the Isaac Gym asset
 * (URDF document order). No reordering is applied.
 *
 * Action contract:
 *   a        = clamp(raw_action, +-ClipAction)            (ClipAction = 18 in training)
 *   q_target = default_q + action_scale * a               (action_scale = 0.25 * effort / Kp)
 *   q_target is additionally clamped to [joint_clip_lower, joint_clip_upper] (URDF position
 *   limits). Training does NOT clip q_target; this clamp is a deployment-side protection
 *   kept from the original framework (documented difference, do not remove for real robot).
 *   PD (Kp/Kd from ActionManager MotorProperties) is executed by the motor firmware
 *   (real robot) or MujocoJoint (sim), same as the original framework.
 *
 * Restart handling: the framework has no "task enabled" hook, so the worker detects a
 * restart by itself: if the time gap since the previous run exceeds one policy period
 * (the task list was disabled in between, e.g. before the first 'r' press), the phase
 * counter is reset to 0, last action is zeroed and the next frame is replicated to fill
 * the whole 15-frame history (same as training episode reset). Note: pressing 'p' does
 * NOT disable the inference task in the original framework, so 'p' then 'r' without an
 * intermediate task disable does not reset phase/history (original behaviour preserved).
 */
#pragma once

#include "CommonLocoInferenceWorker.hpp"
#include "NetInferenceWorker.h"
#include "Utils/ZenBuffer.hpp"
#include "Utils/StaticStringUtils.hpp"
#include "Utils/so3.hpp"
#include <chrono>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace z
{
    template<typename SchedulerType, CTString NetName, typename InferencePrecision, size_t INPUT_STUCK_LENGTH, size_t JOINT_NUMBER>
    class HumanoidGymAmpInferenceWorker : public CommonLocoInferenceWorker<SchedulerType, NetName, InferencePrecision, JOINT_NUMBER>
    {
    public:
        using MotorValVec = math::Vector<InferencePrecision, JOINT_NUMBER>;
        using ValVec3 = math::Vector<InferencePrecision, 3>;
        using ValVec4 = math::Vector<InferencePrecision, 4>;

    public:
        HumanoidGymAmpInferenceWorker(SchedulerType::Ptr scheduler, const nlohmann::json& Net_cfg, const nlohmann::json& Motor_cfg)
            :CommonLocoInferenceWorker<SchedulerType, NetName, InferencePrecision, JOINT_NUMBER>(scheduler, Net_cfg, Motor_cfg),
            HistoryInputBuffer(INPUT_STUCK_LENGTH)
        {
            nlohmann::json NetworkCfg = Net_cfg["Network"];
            nlohmann::json PreprocessCfg = Net_cfg["Preprocess"];

            this->cycle_time_ = NetworkCfg["Cycle_time"].get<InferencePrecision>();
            this->policy_dt_ = NetworkCfg["PolicyDt"].get<InferencePrecision>();
            this->waist_compensation_ = PreprocessCfg.value("WaistCompensation", true);

            const InferencePrecision spin_dt = static_cast<InferencePrecision>(scheduler->getSpinOnceTime());
            this->expected_div_ticks_ = static_cast<size_t>(std::llround(this->policy_dt_ / spin_dt));
            if (this->expected_div_ticks_ == 0)
                throw(std::runtime_error("HumanoidGymAmpInferenceWorker: PolicyDt is smaller than scheduler dt!"));

            this->PrintSplitLine();
            std::cout << "HumanoidGymAmpInferenceWorker (AMP v153 projected-gravity)" << std::endl;
            std::cout << "JOINT_NUMBER=" << JOINT_NUMBER << std::endl;
            std::cout << "INPUT_STUCK_LENGTH=" << INPUT_STUCK_LENGTH << std::endl;
            std::cout << "INPUT_TENSOR_LENGTH=" << INPUT_TENSOR_LENGTH << std::endl;
            std::cout << "Cycle_time=" << this->cycle_time_ << std::endl;
            std::cout << "PolicyDt=" << this->policy_dt_ << std::endl;
            std::cout << "ExpectedDivTicks=" << this->expected_div_ticks_ << std::endl;
            std::cout << "WaistCompensation=" << this->waist_compensation_ << std::endl;
            this->PrintSplitLine();

            // training command scale is fixed: [vx, vy, yaw_rate] * [2, 2, 1]
            auto clock_scales = math::Vector<InferencePrecision, 2>::ones();
            ValVec3 cmd_scales = { static_cast<InferencePrecision>(2.0), static_cast<InferencePrecision>(2.0), static_cast<InferencePrecision>(1.0) };
            this->InputScaleVec = math::cat(
                clock_scales,
                cmd_scales,
                this->Scales_dof_pos,
                this->Scales_dof_vel,
                this->Scales_last_action,
                this->Scales_ang_vel,
                this->Scales_project_gravity
            );
            this->OutputScaleVec = this->ActionScale;

            //wrap input tensor
            this->InputOrtTensors__.push_back(this->WarpOrtTensor(InputTensor));
            this->OutputOrtTensors__.push_back(this->WarpOrtTensor(OutputTensor));
        }

        virtual ~HumanoidGymAmpInferenceWorker()
        {
        }

        void PreProcess() override
        {
            this->start_time = std::chrono::steady_clock::now();

            MotorValVec CurrentMotorPos;
            this->Scheduler->template GetData<"CurrentMotorPosition">(CurrentMotorPos);

            MotorValVec CurrentMotorVel;
            this->Scheduler->template GetData<"CurrentMotorVelocity">(CurrentMotorVel);

            MotorValVec LastAction;
            this->Scheduler->template GetData<concat(NetName, "NetLastAction")>(LastAction);

            ValVec3 UserCmd3;
            this->Scheduler->template GetData<concat(NetName, "NetUserCommand3")>(UserCmd3);

            ValVec3 AngVelTorso;
            this->Scheduler->template GetData<"AngleVelocityValue">(AngVelTorso);

            ValVec3 RpyTorso;
            this->Scheduler->template GetData<"AngleValue">(RpyTorso);

            //restart detection: the inference task was disabled since the last run
            //(e.g. before the first policy start), so policy state must be reset.
            const size_t now = this->Scheduler->getTimeStamp();
            if (!this->has_run_ || (now - this->last_timestamp_) > this->expected_div_ticks_)
            {
                this->ResetPolicyState();
                LastAction.fill(static_cast<InferencePrecision>(0));
            }
            this->last_timestamp_ = now;

            //phase clock: phase = step * PolicyDt / Cycle_time, reset to 0 at policy (re)start
            const InferencePrecision phase =
                static_cast<InferencePrecision>(this->phase_step_) * this->policy_dt_ / this->cycle_time_;
            math::Vector<InferencePrecision, 2> ClockVector = {
                std::sin(phase * static_cast<InferencePrecision>(2.0 * M_PI)),
                std::cos(phase * static_cast<InferencePrecision>(2.0 * M_PI))
            };
            this->phase_step_++;

            //map torso-frame IMU data into the pelvis (training base) frame
            ValVec3 ProjectedGravity;
            ValVec3 BaseAngVel;
            if constexpr (JOINT_NUMBER >= 15)
            {
                if (this->waist_compensation_)
                {
                    this->WaistCompensate(CurrentMotorPos, CurrentMotorVel, RpyTorso, AngVelTorso,
                        ProjectedGravity, BaseAngVel);
                }
                else
                {
                    ProjectedGravity = ComputeProjectedGravity(RpyTorso);
                    BaseAngVel = AngVelTorso;
                }
            }
            else
            {
                ProjectedGravity = ComputeProjectedGravity(RpyTorso);
                BaseAngVel = AngVelTorso;
            }

            CurrentMotorPos -= this->JointDefaultPos;

            auto SingleInputVecScaled = math::cat(
                ClockVector,
                UserCmd3,
                CurrentMotorPos,
                CurrentMotorVel,
                LastAction,
                BaseAngVel,
                ProjectedGravity
            ) * this->InputScaleVec;

            if (this->fill_history_)
            {
                //first frame after (re)start is replicated to fill the whole history,
                //same as the training episode reset
                for (size_t i = 0; i < INPUT_STUCK_LENGTH; i++)
                    this->HistoryInputBuffer.push(SingleInputVecScaled);
                this->fill_history_ = false;
            }
            else
            {
                this->HistoryInputBuffer.push(SingleInputVecScaled);
            }

            math::Vector<InferencePrecision, INPUT_TENSOR_LENGTH> InputVec;
            for (size_t i = 0; i < INPUT_STUCK_LENGTH; i++)
            {
                //RingBuffer index 0 is the oldest frame: oldest -> newest
                std::copy(this->HistoryInputBuffer[i].begin(), this->HistoryInputBuffer[i].end(), InputVec.begin() + i * INPUT_TENSOR_LENGTH_UNIT);
            }

            this->InputTensor.Array() = z::math::clamp(InputVec, -this->ClipObservation, this->ClipObservation);
        }

        void PostProcess() override
        {
            auto LastAction = this->OutputTensor.toVector();
            auto ClipedLastAction = z::math::clamp(LastAction, -this->ClipAction, this->ClipAction);
            this->Scheduler->template SetData<concat(NetName, "NetLastAction")>(ClipedLastAction);

            auto ScaledAction = ClipedLastAction * this->OutputScaleVec + this->JointDefaultPos;

            //deployment-side joint position protection (URDF limits); training does not clip q_target
            auto clipedAction = z::math::clamp(ScaledAction, this->JointClipLower, this->JointClipUpper);
            this->Scheduler->template SetData<concat(NetName, "Action")>(clipedAction);

            this->end_time = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(this->end_time - this->start_time);
            InferencePrecision inference_time = static_cast<InferencePrecision>(duration.count());
            this->Scheduler->template SetData<concat(NetName, "InferenceTime")>(inference_time);
        }

    private:
        /**
         * @brief reset phase/history/last-action, called when a policy (re)start is detected
         */
        void ResetPolicyState()
        {
            this->phase_step_ = 0;
            this->fill_history_ = true;
            this->has_run_ = true;

            MotorValVec zero_action = MotorValVec::zeros();
            this->Scheduler->template SetData<concat(NetName, "NetLastAction")>(zero_action);

            std::cout << "[HumanoidGymAmp] policy state reset: phase=0, last_action=0, history will be refilled" << std::endl;
        }

        /**
         * @brief map torso-frame IMU measurements into the pelvis (training base) frame
         * using the waist joint chain pelvis -> yaw(z) -> roll(x) -> pitch(y) -> torso.
         * Waist device indices are 12 (yaw), 13 (roll), 14 (pitch) in device/training order.
         */
        void WaistCompensate(const MotorValVec& q, const MotorValVec& dq,
            ValVec3 rpy_torso, const ValVec3& gyro_torso,
            ValVec3& projected_gravity_pelvis, ValVec3& ang_vel_pelvis) const
        {
            constexpr size_t kWaistYaw = 12;
            constexpr size_t kWaistRoll = 13;
            constexpr size_t kWaistPitch = 14;

            const InferencePrecision q_yaw = q[kWaistYaw];
            const InferencePrecision q_roll = q[kWaistRoll];
            const InferencePrecision q_pitch = q[kWaistPitch];

            const ValVec4 quat_z = AxisQuatZ(q_yaw);
            const ValVec4 quat_x = AxisQuatX(q_roll);
            const ValVec4 quat_y = AxisQuatY(q_pitch);

            //gravity: g_pelvis = Rz(q_yaw) * Rx(q_roll) * Ry(q_pitch) * g_torso
            const ValVec3 g_torso = ComputeProjectedGravity(rpy_torso);
            projected_gravity_pelvis = math::quat_rotate(quat_z,
                math::quat_rotate(quat_x,
                    math::quat_rotate(quat_y, g_torso)));

            //relative angular velocity of the torso w.r.t. the pelvis, expressed in the torso frame.
            //Joint axes mapped into the torso frame (transpose of the pelvis->torso rotation):
            //  yaw axis:  Ry(-q_pitch) * Rx(-q_roll) * z
            //  roll axis: Ry(-q_pitch) * x
            //  pitch axis: y
            //(verified numerically against MuJoCo FK to 3e-15 over random configurations)
            const ValVec4 quat_y_inv = AxisQuatY(-q_pitch);
            const ValVec4 quat_x_inv = AxisQuatX(-q_roll);
            const ValVec3 axis_z = { 0, 0, 1 };
            const ValVec3 axis_x = { 1, 0, 0 };
            const ValVec3 axis_y = { 0, 1, 0 };
            const ValVec3 omega_rel =
                math::quat_rotate(quat_y_inv, math::quat_rotate(quat_x_inv, axis_z)) * dq[kWaistYaw] +
                math::quat_rotate(quat_y_inv, axis_x) * dq[kWaistRoll] +
                axis_y * dq[kWaistPitch];

            //omega_pelvis = Rz * Rx * Ry * (omega_torso - omega_rel)
            const ValVec3 omega_diff = gyro_torso - omega_rel;
            ang_vel_pelvis = math::quat_rotate(quat_z,
                math::quat_rotate(quat_x,
                    math::quat_rotate(quat_y, omega_diff)));
        }

        static ValVec4 AxisQuatX(InferencePrecision angle)
        {
            const InferencePrecision h = angle * static_cast<InferencePrecision>(0.5);
            return { std::sin(h), 0, 0, std::cos(h) }; //XYZW
        }
        static ValVec4 AxisQuatY(InferencePrecision angle)
        {
            const InferencePrecision h = angle * static_cast<InferencePrecision>(0.5);
            return { 0, std::sin(h), 0, std::cos(h) }; //XYZW
        }
        static ValVec4 AxisQuatZ(InferencePrecision angle)
        {
            const InferencePrecision h = angle * static_cast<InferencePrecision>(0.5);
            return { 0, 0, std::sin(h), std::cos(h) }; //XYZW
        }

    private:
        //clock(2); usercmd(3); q; dq; last action; base ang vel(3); projected gravity(3)
        static constexpr size_t INPUT_TENSOR_LENGTH_UNIT = 2 + 3 + JOINT_NUMBER + JOINT_NUMBER + JOINT_NUMBER + 3 + 3;
        static constexpr size_t INPUT_TENSOR_LENGTH = INPUT_TENSOR_LENGTH_UNIT * INPUT_STUCK_LENGTH;
        static constexpr size_t OUTPUT_TENSOR_LENGTH = JOINT_NUMBER;

        //input tensor
        z::math::Tensor<InferencePrecision, 1, INPUT_TENSOR_LENGTH> InputTensor;
        z::math::Vector<InferencePrecision, INPUT_TENSOR_LENGTH_UNIT> InputScaleVec;
        z::RingBuffer<z::math::Vector<InferencePrecision, INPUT_TENSOR_LENGTH_UNIT>> HistoryInputBuffer;

        //output tensor
        z::math::Tensor<InferencePrecision, 1, OUTPUT_TENSOR_LENGTH> OutputTensor;
        z::math::Vector<InferencePrecision, OUTPUT_TENSOR_LENGTH> OutputScaleVec;

        //gait cycle time and policy dt
        InferencePrecision cycle_time_;
        InferencePrecision policy_dt_;

        //policy state (reset on restart)
        size_t phase_step_ = 0;
        size_t last_timestamp_ = 0;
        size_t expected_div_ticks_ = 1;
        bool has_run_ = false;
        bool fill_history_ = true;
        bool waist_compensation_ = true;

        //compute time
        std::chrono::steady_clock::time_point start_time;
        std::chrono::steady_clock::time_point end_time;
    };
};
