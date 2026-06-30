#include "State_Mimic.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/utils/utils.h"

static Eigen::Quaternionf init_quat = Eigen::Quaternionf::Identity();
std::shared_ptr<State_Mimic::MotionLoader_> State_Mimic::motion = nullptr;


// Helper: motion pelvis → torso quat (pelvis * Rz(waist_yaw))
// TORSO_JOINT_IDX defined in unitree_articulation.h
Eigen::Quaternionf motion_anchor_quat_w(std::shared_ptr<State_Mimic::MotionLoader_> loader)
{
    const auto root_quat = loader->root_quaternion();
    const auto joint_pos = loader->joint_pos();
    if (joint_pos.size() <= unitree::TORSO_JOINT_IDX) {
        throw std::runtime_error(
            "motion_anchor_quat_w: motion data has " + std::to_string(joint_pos.size()) +
            " joints, need at least " + std::to_string(unitree::TORSO_JOINT_IDX + 1));
    }
    Eigen::Quaternionf torso_quat = root_quat
        * Eigen::AngleAxisf(joint_pos[unitree::TORSO_JOINT_IDX], Eigen::Vector3f::UnitZ());

    return torso_quat;
}


// SKILL 开环注入: 初始化 replay 所需的静态变量
void State_Mimic::init_replay(std::shared_ptr<MotionLoader_> motion_ptr,
                               const Eigen::Quaternionf& ref_quat,
                               const Eigen::Quaternionf& robot_quat)
{
    motion = motion_ptr;
    // 使用 yawQuaternion() 而非手动 atan2+AngleAxis 以对齐 State_Mimic::enter() 和 Python
    // 原因: 手动 atan2(R(1,0), R(0,0)) 虽然数学等价于 yawQuaternion(), 但 float32 舍入路径不同,
    // 会产出 ~2e-3 的 init_quat 差异, 并级联放大到 motion_anchor_ori_b 的 ~2.87e-3 误差
    //
    // NOTE: ref_quat MUST be the motion TORSO quat (pelvis * Rz(waist_yaw)),
    //       robot_quat MUST be the robot TORSO quat (from IMU on torso).
    //       Passing pelvis quats will produce a wrong yaw alignment off by Rz(waist_yaw).
    auto ref_yaw_quat = isaaclab::yawQuaternion(ref_quat);
    auto robot_yaw_quat = isaaclab::yawQuaternion(robot_quat);
    init_quat = robot_yaw_quat * ref_yaw_quat.conjugate();
}


namespace isaaclab
{
namespace mdp
{

REGISTER_OBSERVATION(motion_command)
{
    auto loader = State_Mimic::motion;
    std::vector<float> data;

    auto motion_joint_pos = loader->joint_pos();
    auto motion_joint_vel = loader->joint_vel();

    data.insert(data.end(),
                motion_joint_pos.data(),
                motion_joint_pos.data() + motion_joint_pos.size());
    data.insert(data.end(),
                motion_joint_vel.data(),
                motion_joint_vel.data() + motion_joint_vel.size());
    return data;
}

REGISTER_OBSERVATION(motion_anchor_ori_b)
{
    auto loader = State_Mimic::motion;

    // BUGFIX: Use motion TORSO quat (pelvis * Rz(torso_joint)) to match training.
    // Training side uses body_link_quat_w[:, torso_body_id] which is the torso frame.
    // Previously using loader->root_quaternion() (pelvis frame) produced a relative
    // orientation off by Rz(motion_waist_yaw), contaminating the 6D orientation obs.
    auto real_quat_w = env->robot->data.root_quat_w;   // robot torso (IMU)

    // Bounds check: motion data must have at least TORSO_JOINT_IDX + 1 joints
    auto motion_jp = loader->joint_pos();
    if (motion_jp.size() <= unitree::TORSO_JOINT_IDX) {
        return std::vector<float>(6, 0.0f); // safe fallback
    }
    auto ref_quat_w  = loader->root_quaternion()        // motion torso: pelvis * Rz(waist)
        * Eigen::AngleAxisf(motion_jp[unitree::TORSO_JOINT_IDX], Eigen::Vector3f::UnitZ());

    auto rot_ = (init_quat * ref_quat_w).conjugate() * real_quat_w;
    // 必须使用显式类型 Eigen::Matrix3f, 不能用 auto!
    // Eigen lazy evaluation: toRotationMatrix() 返回临时 Matrix3f 对象,
    // .transpose() 返回 Transpose<Matrix3f> 表达式(引用临时对象)。
    // 用 auto 推导为 Transpose<Matrix3f>, 语句结束后临时对象销毁,
    // rot 持有悬空引用 → 所有元素读为 0/垃圾值
    Eigen::Matrix3f rot = rot_.toRotationMatrix().transpose();

    Eigen::Matrix<float, 6, 1> data;
    data << rot(0, 0), rot(0, 1), rot(1, 0), rot(1, 1), rot(2, 0), rot(2, 1);
    return std::vector<float>(data.data(), data.data() + data.size());
}

}
}


State_Mimic::State_Mimic(int state_mode, std::string state_string)
: FSMState(state_mode, state_string) 
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    auto articulation = std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate);

    std::filesystem::path motion_file = cfg["motion_file"].as<std::string>();
    if(!motion_file.is_absolute()) {
        motion_file = param::proj_dir / motion_file;
    }

    // Motion
    motion_ = std::make_shared<MotionLoader_>(motion_file.string());
    spdlog::info("Loaded motion file '{}' with duration {:.2f}s", motion_file.stem().string(), motion_->duration);
    motion = motion_;
    if(cfg["time_start"]) {
        float time_start = cfg["time_start"].as<float>();
        time_range_[0] = std::clamp(time_start, 0.0f, motion_->duration);
    } else {
        time_range_[0] = 0.0f;
    }
    if(cfg["time_end"]) {
        float time_end = cfg["time_end"].as<float>();
        time_range_[1] = std::clamp(time_end, 0.0f, motion_->duration);
    } else {
        time_range_[1] = motion_->duration;
    }

    if (time_range_[0] >= time_range_[1]) {
        throw std::runtime_error(
            "Invalid time_range: time_start (" + std::to_string(time_range_[0]) +
            ") >= time_end (" + std::to_string(time_range_[1]) + ")"
        );
    }

    // Debug: bypass policy and send reference joint positions directly
    if (cfg["debug_ref_action"] && cfg["debug_ref_action"].as<bool>()) {
        debug_ref_action_ = true;
        spdlog::warn("DEBUG MODE ENABLED: joint commands will follow reference motion directly (policy bypassed)");
    }

    std::string end_state = "FixStand";
    if (cfg["end_state"]) {
        end_state = cfg["end_state"].as<std::string>();
    }

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        articulation
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return (env->episode_length * env->step_dt) > time_range_[1]; }, // time out
            FSMStringMap.right.at(end_state)
        )
    );
    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{
                // Compute bad_orientation from raw lowstate data in the FSM thread
                // (pre_run() calls lowstate->update(), so IMU/motor data is fresh).
                // This decouples fall detection from the policy thread's robot->update().
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
}

void State_Mimic::enter()
{
    // Check if dump is enabled
    auto fsm_cfg = param::config["FSM"][getStateString()];
    if (fsm_cfg["dump_enabled"] && fsm_cfg["dump_enabled"].as<bool>()) {
        dump_enabled_ = true;
        std::string dump_path = "/tmp/mimic_dump.csv";
        dump_file_.open(dump_path, std::ios::out | std::ios::trunc);
        // Write header
        dump_file_ << "step,episode_length";
        for (int i = 0; i < 144; i++) dump_file_ << ",obs_" << i;
        for (int i = 0; i < 27; i++) dump_file_ << ",raw_action_" << i;
        for (int i = 0; i < 27; i++) dump_file_ << ",processed_action_" << i;
        for (int i = 0; i < 27; i++) dump_file_ << ",motor_q_" << i;
        for (int i = 0; i < 27; i++) dump_file_ << ",motor_dq_" << i;
        dump_file_ << "\n";
        spdlog::info("Mimic dump enabled, writing to {}", dump_path);
    }

    // set gain
    for (int i = 0; i < env->robot->data.joint_stiffness.size(); i++)
    {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].kp() = env->robot->data.joint_stiffness[i];
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].kd() = env->robot->data.joint_damping[i];
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].dq() = 0;
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].tau() = 0;
    }

    motion = motion_; // set for specific motion
    // NOTE: env->reset() is deferred to the policy thread after motion initialization,
    // so that motion-dependent observations (motion_command, motion_anchor_ori_b) are
    // computed with valid reference data during reset.

    // Start policy thread
    policy_thread_running = true;
    policy_thread = std::thread([this]{
        using clock = std::chrono::high_resolution_clock;
        const std::chrono::duration<double> desiredDuration(env->step_dt);
        const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

        // Initialize timing
        const auto start = clock::now();
        auto sleepTill = start + dt;

        // 1. Update robot data first so motion->reset() has fresh data
        env->robot->update();

        // 2. Initialize motion reference at start time
        motion->reset(env->robot->data, time_range_[0]);
        // BUGFIX: Use motion TORSO quat (pelvis * Rz(torso_joint)) for yaw alignment,
        // matching training which uses body_link_quat_w[:, torso_body_id].
        // Previously using root_quaternion() (pelvis yaw) introduced Rz(waist_yaw)
        // error in init_quat, which cascaded into motion_anchor_ori_b observations.
        auto motion_jp = motion->joint_pos();
        if (motion_jp.size() <= unitree::TORSO_JOINT_IDX) {
            spdlog::error("init_quat: motion data has {} joints, need >= {}", 
                          motion_jp.size(), unitree::TORSO_JOINT_IDX + 1);
            init_quat = Eigen::Quaternionf::Identity(); // safe fallback
        } else {
            auto motion_torso_quat = motion->root_quaternion()
                * Eigen::AngleAxisf(motion_jp[unitree::TORSO_JOINT_IDX], Eigen::Vector3f::UnitZ());
            auto ref_yaw_quat = isaaclab::yawQuaternion(motion_torso_quat);
            auto robot_yaw_quat = isaaclab::yawQuaternion(env->robot->data.root_quat_w);
            init_quat = robot_yaw_quat * ref_yaw_quat.conjugate();
        }

        // 3. Align default_joint_pos to current robot pose, so that joint_pos_rel ≈ 0
        //    and the policy sees a consistent starting state regardless of entry source.
        env->robot->data.default_joint_pos = env->robot->data.joint_pos;

        // 3. Compute the initial action that maps to the motion reference position.
        //    This ensures last_action reflects the actual prior command for temporal models,
        //    rather than being zeroed (which would be inconsistent with the robot's pose).
        auto action_scale = env->cfg["actions"]["JointPositionAction"]["scale"].as<std::vector<float>>();
        auto action_offset = env->cfg["actions"]["JointPositionAction"]["offset"].as<std::vector<float>>();
        auto motion_joint_pos = motion->joint_pos();
        std::vector<float> initial_action(motion_joint_pos.size());
        for(size_t i = 0; i < motion_joint_pos.size(); ++i) {
            initial_action[i] = (motion_joint_pos[i] - action_offset[i]) / action_scale[i];
            initial_action[i] = std::clamp(initial_action[i], -1.5f, 1.5f);
        }

        // 4. Reset env: fills observation history (joint_pos_rel=0, motion obs correct)
        //    then override last_action with the computed initial action.
        env->reset();
        env->action_manager->set_action(initial_action);
        env->observation_manager->recompute_term_history("last_action");
        env->action_manager->process_action(initial_action);

        while (policy_thread_running)
        {
            env->robot->update();
            motion->update(env->episode_length * env->step_dt + time_range_[0]);
            env->step();

            // Write dump data
            if (this->dump_enabled_ && this->dump_file_.is_open()) {
                auto& fout = this->dump_file_;
                fout << env->episode_length << ","
                     << env->episode_length;

                // Observation (144 dims)
                for (size_t i = 0; i < env->last_observation.size() && i < 144; i++) {
                    fout << "," << env->last_observation[i];
                }
                for (size_t i = env->last_observation.size(); i < 144; i++) {
                    fout << ",0";
                }

                // Raw action (27 dims)
                for (size_t i = 0; i < env->last_raw_action.size() && i < 27; i++) {
                    fout << "," << env->last_raw_action[i];
                }
                for (size_t i = env->last_raw_action.size(); i < 27; i++) {
                    fout << ",0";
                }

                // Processed action (27 dims)
                auto processed = env->action_manager->processed_actions();
                for (size_t i = 0; i < processed.size() && i < 27; i++) {
                    fout << "," << processed[i];
                }
                for (size_t i = processed.size(); i < 27; i++) {
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

            // Sleep
            std::this_thread::sleep_until(sleepTill);
            sleepTill += dt;
        }
    });
}


void State_Mimic::run()
{
    if (debug_ref_action_) {
        // DEBUG MODE: Send reference joint positions directly, bypassing policy
        auto ref_pos = motion_->joint_pos();
        auto& joint_ids = env->robot->data.joint_ids_map;
        for (size_t i = 0; i < joint_ids.size() && i < static_cast<size_t>(ref_pos.size()); i++) {
            lowcmd->msg_.motor_cmd()[joint_ids[i]].q() = ref_pos[i];
        }
    } else {
        auto action = env->action_manager->processed_actions();
        for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
            lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
        }
    }
}
