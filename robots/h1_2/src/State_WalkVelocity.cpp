// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#include "FSM/State_WalkVelocity.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/terminations.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/utils/utils.h"
#include <Eigen/Geometry>
#include <algorithm>

using namespace std::chrono_literals;


// H1_2 特殊处理：IMU 在 torso 上，需要变换到 pelvis frame
// 参考 h1_2_walk_deploy_real/common/rotation_helper.py
namespace {

// 获取 pelvis 坐标系下的重力向量（考虑 waist_yaw）
Eigen::Vector3f get_projected_gravity_pelvis(isaaclab::ManagerBasedRLEnv* env)
{
    using H1_2Type = unitree::BaseArticulation<LowState_t::SharedPtr>;
    H1_2Type* robot = dynamic_cast<H1_2Type*>(env->robot.get());

    auto & motors = robot->lowstate->msg_.motor_state();
    float waist_yaw = motors[12].q();  // torso_yaw joint index = 12

    // IMU quaternion (w, x, y, z)
    auto & imu = robot->lowstate->msg_.imu_state();
    Eigen::Quaternionf torso_quat(
        imu.quaternion()[0],  // w
        imu.quaternion()[1],  // x
        imu.quaternion()[2],  // y
        imu.quaternion()[3]   // z
    );

    // torso -> pelvis 变换：R_pelvis = R_torso * RzWaist^T
    Eigen::AngleAxisf waist_rotation(waist_yaw, Eigen::Vector3f::UnitZ());
    Eigen::Quaternionf pelvis_quat = torso_quat * waist_rotation.inverse();

    // 投影重力向量
    Eigen::Vector3f gravity_world(0.0f, 0.0f, -1.0f);
    Eigen::Vector3f gravity_pelvis = pelvis_quat.conjugate() * gravity_world;

    return gravity_pelvis;
}

// 获取 pelvis 坐标系下的角速度（考虑 waist_yaw）
Eigen::Vector3f get_ang_vel_pelvis(isaaclab::ManagerBasedRLEnv* env)
{
    using H1_2Type = unitree::BaseArticulation<LowState_t::SharedPtr>;
    H1_2Type* robot = dynamic_cast<H1_2Type*>(env->robot.get());

    auto & motors = robot->lowstate->msg_.motor_state();
    float waist_yaw = motors[12].q();
    float waist_yaw_omega = motors[12].dq();

    // IMU angular velocity
    auto & imu = robot->lowstate->msg_.imu_state();
    Eigen::Vector3f torso_omega(
        imu.gyroscope()[0],
        imu.gyroscope()[1],
        imu.gyroscope()[2]
    );

    // torso -> pelvis 变换
    Eigen::AngleAxisf waist_rotation(waist_yaw, Eigen::Vector3f::UnitZ());
    Eigen::Vector3f pelvis_omega = waist_rotation.toRotationMatrix() * torso_omega;
    pelvis_omega[2] -= waist_yaw_omega;  // 减去 waist 角速度

    return pelvis_omega;
}

} // anonymous namespace


namespace isaaclab
{
namespace mdp
{

// 重写 projected_gravity 观测，考虑 torso IMU -> pelvis 变换
REGISTER_OBSERVATION(projected_gravity_walk)
{
    auto gravity_pelvis = get_projected_gravity_pelvis(env);
    return std::vector<float>(gravity_pelvis.data(), gravity_pelvis.data() + 3);
}

// 重写 base_ang_vel 观测，考虑 torso IMU -> pelvis 变换
REGISTER_OBSERVATION(base_ang_vel_walk)
{
    auto ang_vel_pelvis = get_ang_vel_pelvis(env);
    return std::vector<float>(ang_vel_pelvis.data(), ang_vel_pelvis.data() + 3);
}

}
}


State_WalkVelocity::State_WalkVelocity(int state_mode, std::string state_string)
: FSMState(state_mode, state_string)
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    // 创建 Articulation，仅使用腿部关节 (0-11)
    auto articulation = std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate);

    // 加载环境配置
    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        articulation
    );

    // 加载 ONNX 策略
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    // 加载手臂/腰部固定目标配置
    // 参考 h1_2_walk_deploy_real/configs/h1_2.yaml
    arm_waist_joint_idx_ = {12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26};
    arm_waist_kp_ = {300, 120, 120, 120, 80, 80, 80, 80, 120, 120, 120, 80, 80, 80, 80};
    arm_waist_kd_ = {3, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1};
    arm_waist_target_ = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    // 从 YAML 加载手臂/腰部配置（如果存在）
    if (cfg["arm_waist_target"]) {
        arm_waist_target_ = cfg["arm_waist_target"].as<std::vector<float>>();
    }
    if (cfg["arm_waist_kp"]) {
        arm_waist_kp_ = cfg["arm_waist_kp"].as<std::vector<float>>();
    }
    if (cfg["arm_waist_kd"]) {
        arm_waist_kd_ = cfg["arm_waist_kd"].as<std::vector<float>>();
    }

    // Register fall detection (bad_orientation) — FSM thread variant.
    // pre_run() calls lowstate->update() so IMU/motor data is fresh.
    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{
                auto & imu = lowstate->msg_.imu_state();
                Eigen::Quaternionf imu_quat(
                    imu.quaternion()[0], imu.quaternion()[1],
                    imu.quaternion()[2], imu.quaternion()[3]
                );
                float waist_yaw = lowstate->msg_.motor_state()[unitree::TORSO_JOINT_IDX].q();
                return isaaclab::mdp::bad_orientation(imu_quat, waist_yaw, 1.0f);
            },
            FSMStringMap.right.at("Passive")
        )
    );

    spdlog::info("State_WalkVelocity initialized with policy_dir: {}", policy_dir.string());
}


void State_WalkVelocity::enter()
{
    // Check if dump is enabled
    auto fsm_cfg = param::config["FSM"][getStateString()];
    if (fsm_cfg["dump_enabled"] && fsm_cfg["dump_enabled"].as<bool>()) {
        dump_enabled_ = true;
        std::string dump_path = "/tmp/wv_dump.csv";
        dump_file_.open(dump_path, std::ios::out | std::ios::trunc);
        // Write header
        dump_file_ << "step,global_phase,episode_length";
        for (int i = 0; i < 47; i++) dump_file_ << ",obs_" << i;
        for (int i = 0; i < 12; i++) dump_file_ << ",raw_action_" << i;
        for (int i = 0; i < 12; i++) dump_file_ << ",processed_action_" << i;
        for (int i = 0; i < 27; i++) dump_file_ << ",motor_q_" << i;
        for (int i = 0; i < 27; i++) dump_file_ << ",motor_dq_" << i;
        dump_file_ << "\n";
        spdlog::info("WalkVelocity dump enabled, writing to {}", dump_path);
    }

    // Read current arm/waist positions from lowstate as targets
    auto & motors = lowstate->msg_.motor_state();
    for (int i = 0; i < arm_waist_joint_idx_.size(); i++)
    {
        int motor_idx = arm_waist_joint_idx_[i];
        arm_waist_target_[i] = motors[motor_idx].q();
    }

    // 保存前一状态的 PD 增益（用于平滑过渡）
    prev_kp_.resize(env->robot->data.joint_ids_map.size());
    prev_kd_.resize(env->robot->data.joint_ids_map.size());
    for (int i = 0; i < env->robot->data.joint_ids_map.size(); i++)
    {
        int motor_idx = env->robot->data.joint_ids_map[i];
        prev_kp_[i] = lowcmd->msg_.motor_cmd()[motor_idx].kp();
        prev_kd_[i] = lowcmd->msg_.motor_cmd()[motor_idx].kd();
    }

    // 设置腿部关节增益（从 deploy.yaml 加载）
    for (int i = 0; i < env->robot->data.joint_ids_map.size(); i++)
    {
        int motor_idx = env->robot->data.joint_ids_map[i];
        lowcmd->msg_.motor_cmd()[motor_idx].kp() = env->robot->data.joint_stiffness[i];
        lowcmd->msg_.motor_cmd()[motor_idx].kd() = env->robot->data.joint_damping[i];
        lowcmd->msg_.motor_cmd()[motor_idx].q() = lowstate->msg_.motor_state()[motor_idx].q();
        lowcmd->msg_.motor_cmd()[motor_idx].dq() = 0;
        lowcmd->msg_.motor_cmd()[motor_idx].tau() = 0;
    }

    // 设置手臂/腰部关节增益和目标
    for (int i = 0; i < arm_waist_joint_idx_.size(); i++)
    {
        int motor_idx = arm_waist_joint_idx_[i];
        lowcmd->msg_.motor_cmd()[motor_idx].q() = arm_waist_target_[i];
        lowcmd->msg_.motor_cmd()[motor_idx].dq() = 0;
        lowcmd->msg_.motor_cmd()[motor_idx].kp() = arm_waist_kp_[i];
        lowcmd->msg_.motor_cmd()[motor_idx].kd() = arm_waist_kd_[i];
        lowcmd->msg_.motor_cmd()[motor_idx].tau() = 0;
    }

    env->robot->update();

    // 初始化 PD 平滑过渡参数
    enter_time_ = std::chrono::steady_clock::now();
    pd_transition_active_ = true;

    // 启动策略线程
    policy_thread_running = true;
    policy_thread = std::thread([this]{
        using clock = std::chrono::high_resolution_clock;
        const std::chrono::duration<double> desiredDuration(env->step_dt);
        const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

        auto sleepTill = clock::now() + dt;

        // 1. Keep training default_joint_pos (from deploy.yaml).
        //    Do NOT override with current_pos. Overriding causes two problems:
        //    (a) joint_pos_rel=0 tells the policy "robot is at training default pose",
        //        but the actual PD targets (computed from raw_action * scale + offset)
        //        are in training-default-space. The mismatch creates huge PD errors.
        //    (b) Even with zero initial_action, the PD error feedback loop causes
        //        raw_action to grow over time (observed: 3.4→5.0→5.4 in 3 steps).
        //    Using training default_joint_pos lets the policy see the real offset
        //    and output appropriate corrective actions.
        env->robot->update();

        // 2. Initialize last_action to zero vector.
        //    This matches training-time initialization (actions = zeros at episode start).
        //    Computing inverse action (pos-offset)/scale creates an observation contradiction:
        //    joint_pos_rel=0 (robot at default pose) but last_action=huge (contradicts "at rest").
        //    The contradictory observation causes ONNX to output extreme values (e.g. -3.75),
        //    leading to violent shaking and robot collapse.
        std::vector<float> initial_action(env->robot->data.joint_pos.size(), 0.0f);

        // 3. Reset env: fills observation history with correct joint_pos_rel
        //    (relative to training default_joint_pos), then set last_action to zero.
        env->reset();
        env->action_manager->set_action(initial_action);
        env->observation_manager->recompute_term_history("last_action");
        env->action_manager->process_action(initial_action);

        // ---- [DIAG] Dump enter() state ----
        printf("[DIAG] Enter WalkVelocity\n");
        printf("[DIAG] episode_length=%ld global_phase=%.6f step_dt=%.4f\n",
               env->episode_length, env->global_phase, env->step_dt);
        printf("[DIAG] default_joint_pos:");
        for (int i = 0; i < env->robot->data.default_joint_pos.size(); i++)
            printf(" %.4f", env->robot->data.default_joint_pos[i]);
        printf("\n");
        printf("[DIAG] initial_action (raw):");
        for (auto v : initial_action) printf(" %.4f", v);
        printf("\n");
        printf("[DIAG] initial_action is zero vector (matches training init)\n");
        // ---- [DIAG END] ----

        // 4. PRE-STABILIZATION: Wait for robot to physically reach training default_joint_pos.
        //    During this period, run() reads processed_actions (which = default_joint_pos
        //    from process_action(zero) = 0*scale+offset = offset = default_joint_pos).
        //    The PD holds at default, allowing the robot to settle upright.
        //    Without this, the first policy step sees joint_pos_rel ≈ 0 but
        //    the robot is actually tilted (projected_gravity far from (0,0,-1)),
        //    causing the ONNX strategy to output extreme corrective actions.
        const int settle_steps = 25;  // 0.5 seconds at 50 Hz
        printf("[DIAG] Settling for %d steps (%.1f sec) at default_joint_pos...\n",
               settle_steps, settle_steps * env->step_dt);
        for (int i = 0; i < settle_steps; i++) {
            std::this_thread::sleep_until(sleepTill);
            sleepTill += dt;
        }
        printf("[DIAG] Settling complete. Starting policy.\n");

        int step_counter = 0;

        while (policy_thread_running)
        {
            // PD 增益平滑过渡（从 FixStand 增益渐变到 WalkVelocity 增益）
            if (pd_transition_active_) {
                auto elapsed = std::chrono::steady_clock::now() - enter_time_;
                float t = std::chrono::duration<float>(elapsed).count() / pd_transition_duration_;
                t = std::clamp(t, 0.0f, 1.0f);
                for (size_t i = 0; i < env->robot->data.joint_ids_map.size(); i++) {
                    int motor_idx = env->robot->data.joint_ids_map[i];
                    float target_kp = env->robot->data.joint_stiffness[i];
                    float target_kd = env->robot->data.joint_damping[i];
                    lowcmd->msg_.motor_cmd()[motor_idx].kp() = prev_kp_[i] + t * (target_kp - prev_kp_[i]);
                    lowcmd->msg_.motor_cmd()[motor_idx].kd() = prev_kd_[i] + t * (target_kd - prev_kd_[i]);
                }
                if (t >= 1.0f) pd_transition_active_ = false;
            }

            env->step();

            // ---- [DIAG] First 3 steps dump ----
            if (step_counter < 3) {
                printf("[DIAG] Step %d\n", step_counter);
                printf("[DIAG]   obs (47):");
                for (int i = 0; i < 47 && i < (int)env->last_observation.size(); i++)
                    printf(" %.6f", env->last_observation[i]);
                printf("\n");
                printf("[DIAG]   raw_action (12):");
                for (int i = 0; i < 12 && i < (int)env->last_raw_action.size(); i++)
                    printf(" %.6f", env->last_raw_action[i]);
                printf("\n");
                auto proc = env->action_manager->processed_actions();
                printf("[DIAG]   proc_action (12):");
                for (int i = 0; i < 12 && i < (int)proc.size(); i++)
                    printf(" %.6f", proc[i]);
                printf("\n");
                printf("[DIAG]   global_phase=%.6f episode_length=%ld\n",
                       env->global_phase, env->episode_length);
            }
            // ---- [DIAG END] ----
            step_counter++;

            // Write dump data
            if (this->dump_enabled_ && this->dump_file_.is_open()) {
                auto& fout = this->dump_file_;
                fout << env->episode_length << ","
                     << env->global_phase << ","
                     << env->episode_length;

                // Observation (47 dims)
                for (size_t i = 0; i < env->last_observation.size() && i < 47; i++) {
                    fout << "," << env->last_observation[i];
                }
                // Pad if observation is shorter than 47
                for (size_t i = env->last_observation.size(); i < 47; i++) {
                    fout << ",0";
                }

                // Raw action (12 dims)
                for (size_t i = 0; i < env->last_raw_action.size() && i < 12; i++) {
                    fout << "," << env->last_raw_action[i];
                }
                for (size_t i = env->last_raw_action.size(); i < 12; i++) {
                    fout << ",0";
                }

                // Processed action (12 dims) - from action_manager
                auto processed = env->action_manager->processed_actions();
                for (size_t i = 0; i < processed.size() && i < 12; i++) {
                    fout << "," << processed[i];
                }
                for (size_t i = processed.size(); i < 12; i++) {
                    fout << ",0";
                }

                // Motor positions and velocities (27 each)
                auto& motors = this->lowstate->msg_.motor_state();
                for (int i = 0; i < 27; i++) {
                    fout << "," << motors[i].q();
                }
                for (int i = 0; i < 27; i++) {
                    fout << "," << motors[i].dq();
                }

                fout << "\n";
            }

            std::this_thread::sleep_until(sleepTill);
            sleepTill += dt;
        }
    });
}


void State_WalkVelocity::run()
{
    // 获取策略输出的腿部动作
    auto action = env->action_manager->processed_actions();

    // 写入腿部关节目标
    for (int i = 0; i < env->robot->data.joint_ids_map.size(); i++)
    {
        int motor_idx = env->robot->data.joint_ids_map[i];
        lowcmd->msg_.motor_cmd()[motor_idx].q() = action[i];
    }

    // 手臂/腰部保持固定目标（已在 enter() 中设置）
}