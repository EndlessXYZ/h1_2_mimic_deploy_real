// SKILL Step 3: C++ 侧离线回放验证器 (Mimic 策略)
//
// 读取 Python 采集的 golden_trace.npz，注入原始传感器数据和 motion 数据
// 到真实的 ManagerBasedRLEnv + ObservationManager + ActionManager + OrtRunner 管线，
// 逐帧跑 A→F 全链路，dump 每层输出到 cpp_trace.npz。
//
// 用法: ./replay_validator_mimic <golden_trace.npz> <deploy.yaml> <policy.onnx> <motion.npz> <output_dir>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <memory>

#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include <cnpy.h>

#include "isaaclab/assets/articulation/articulation.h"
#include "isaaclab/envs/manager_based_rl_env.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/algorithms/algorithms.h"
#include "State_Mimic.h"

using namespace isaaclab;

// ============================================================================
// 提供 FSMState 的静态成员定义 (链接器需要, 来自 State_Mimic.h → FSMState.h)
// ============================================================================
#include "Types.h"
std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
std::shared_ptr<Keyboard> FSMState::keyboard = nullptr;

// ============================================================================
// ReplayMotionLoader: 从 motion NPZ 提供 motion 数据 (复用 MotionLoader_ 逻辑)
// ============================================================================
class ReplayMotionLoader : public State_Mimic::MotionLoader_
{
public:
    ReplayMotionLoader(const std::string& motion_file)
        : State_Mimic::MotionLoader_(motion_file) {}
};

// ============================================================================
// ReplayArticulation: 绕过 DDS，直接注入 ArticulationData
// ============================================================================
class ReplayArticulation : public Articulation {
public:
    void update() override {
        // 不调用 BaseArticulation::update()（那需要 DDS lowstate）
        // 数据由外部直接注入到 data.* 中
    }
};

// ============================================================================
// 辅助函数：从 cnpy::NpyArray 读取 float 数据
// ============================================================================
static std::vector<float> read_float_vec(const cnpy::NpyArray& arr) {
    if (arr.word_size == sizeof(float)) {
        return arr.as_vec<float>();
    }
    auto d = arr.as_vec<double>();
    std::vector<float> f(d.size());
    for (size_t i = 0; i < d.size(); ++i) f[i] = static_cast<float>(d[i]);
    return f;
}

static std::vector<int> read_int_vec(const cnpy::NpyArray& arr) {
    // Always read as float first, then convert to int.
    // This handles both float32 and float64 NPZ data correctly.
    // Using arr.as_vec<float>() directly for float32 or reading as double and converting.
    std::vector<float> f;
    if (arr.word_size == sizeof(double)) {
        auto d = arr.as_vec<double>();
        f.resize(d.size());
        for (size_t i = 0; i < d.size(); ++i) f[i] = static_cast<float>(d[i]);
    } else {
        f = arr.as_vec<float>();
    }
    std::vector<int> v(f.size());
    for (size_t i = 0; i < f.size(); ++i) v[i] = static_cast<int>(std::round(f[i]));
    return v;
}

// ============================================================================
// main
// ============================================================================
int main(int argc, char* argv[])
{
    if (argc < 6) {
        std::cerr << "用法: " << argv[0]
                  << " <golden_trace.npz> <deploy.yaml> <policy.onnx> <motion.npz> <output_dir>\n";
        return 1;
    }

    std::string golden_path = argv[1];
    std::string yaml_path   = argv[2];
    std::string onnx_path   = argv[3];
    std::string motion_path = argv[4];
    std::string out_dir     = argv[5];

    // ---- 1. 加载 golden trace ----
    std::cout << "[1] 加载 golden trace: " << golden_path << std::endl;
    auto npz = cnpy::npz_load(golden_path);

    auto raw_imu_quat  = read_float_vec(npz["raw_imu_quat"]);
    auto raw_imu_gyro  = read_float_vec(npz["raw_imu_gyro"]);
    auto raw_motor_q   = read_float_vec(npz["raw_motor_q"]);
    auto raw_motor_dq  = read_float_vec(npz["raw_motor_dq"]);
    auto raw_waist_yaw = read_float_vec(npz["raw_waist_yaw"]);
    auto raw_waist_yaw_omega = read_float_vec(npz["raw_waist_yaw_omega"]);
    auto motion_window_pos = read_int_vec(npz["motion_window_pos"]);
    // Note: v2 golden trace does NOT have init_quat key.
    // init_quat is computed from the first frame below (line ~130).
    auto nn_output_py  = read_float_vec(npz["nn_output"]);
    auto obs_raw       = read_float_vec(npz["obs_raw"]);

    size_t n_steps = npz["raw_imu_quat"].shape[0];
    std::cout << "    步数: " << n_steps << std::endl;
    std::cout << "    motion_window_pos[0..4]: "
              << motion_window_pos[0] << ", "
              << motion_window_pos[1] << ", "
              << motion_window_pos[2] << ", "
              << motion_window_pos[3] << ", "
              << motion_window_pos[4] << std::endl;

    // ---- 2. 加载配置 ----
    std::cout << "[2] 加载配置: " << yaml_path << std::endl;
    YAML::Node cfg = YAML::LoadFile(yaml_path);

    int NUM_JOINTS = cfg["joint_ids_map"].as<std::vector<float>>().size();
    const int OBS_DIM = 144;
    const int ACT_DIM = 27;
    std::cout << "    关节数: " << NUM_JOINTS << std::endl;

    // ---- 3. 加载 Motion 数据并初始化 State_Mimic 静态变量 ----
    std::cout << "[3] 加载 motion 文件: " << motion_path << std::endl;
    auto replay_motion = std::make_shared<ReplayMotionLoader>(motion_path);

    // 使用第一帧的 ref 和 robot quat 来重建 init_quat
    Eigen::Quaternionf ref_pelvis_0 = replay_motion->root_quaternion(); // frame 0
    Eigen::Quaternionf robot_pelvis_0(
        raw_imu_quat[0], raw_imu_quat[1], raw_imu_quat[2], raw_imu_quat[3]);
    State_Mimic::init_replay(replay_motion, ref_pelvis_0, robot_pelvis_0);

    // ---- 4. 创建环境 ----
    std::cout << "[4] 创建环境" << std::endl;

    auto robot = std::make_shared<ReplayArticulation>();
    robot->data.joint_pos.resize(NUM_JOINTS);
    robot->data.joint_vel.resize(NUM_JOINTS);
    robot->data.root_quat_w = Eigen::Quaternionf(1, 0, 0, 0);
    robot->data.root_ang_vel_b = Eigen::Vector3f::Zero();
    robot->data.projected_gravity_b = Eigen::Vector3f(0, 0, -1);
    robot->data.waist_yaw = 0.0f;
    robot->data.waist_yaw_omega = 0.0f;

    ManagerBasedRLEnv env(cfg, robot);
    env.alg = std::make_unique<OrtRunner>(onnx_path);

    // ---- 5. 重置环境 (对齐 State_Mimic::enter()) ----
    std::cout << "[5] 重置环境" << std::endl;

    // 注入第 0 帧 motor 数据
    for (int j = 0; j < NUM_JOINTS; ++j) {
        robot->data.joint_pos[j] = raw_motor_q[j];
        robot->data.joint_vel[j] = raw_motor_dq[j];
    }
    robot->data.root_quat_w = Eigen::Quaternionf(
        raw_imu_quat[0], raw_imu_quat[1], raw_imu_quat[2], raw_imu_quat[3]);
    robot->data.waist_yaw = raw_waist_yaw[0];
    robot->data.waist_yaw_omega = raw_waist_yaw_omega[0];

    // 保持 YAML 中的 default_joint_pos (与 Python ACTION_OFFSET 一致)
    // 注意: 真实 enter() 会覆写为 live motor_q, 但开环校验需要 Python/C++ 使用相同基准
    robot->data.default_joint_vel = Eigen::VectorXf::Zero(NUM_JOINTS);

    env.action_manager->reset();
    env.observation_manager->reset();
    env.reset();
    env.action_manager->set_action(std::vector<float>(ACT_DIM, 0.0f));
    env.observation_manager->recompute_term_history("last_action");
    env.global_phase = 0.0f;
    env.episode_length = 0;

    // ---- 6. 分配输出 buffer ----
    // B/C 层: C++ 独立计算的 6 个观测 term (用于与 golden obs_raw 逐段对比)
    std::vector<float> cpp_obs_motion_command(n_steps * 54);
    std::vector<float> cpp_obs_motion_anchor_ori_b(n_steps * 6);
    std::vector<float> cpp_obs_base_ang_vel(n_steps * 3);
    std::vector<float> cpp_obs_joint_pos(n_steps * 27);
    std::vector<float> cpp_obs_joint_vel(n_steps * 27);
    std::vector<float> cpp_obs_last_action(n_steps * 27);
    // D/F 层
    std::vector<float> cpp_nn_output(n_steps * ACT_DIM);
    std::vector<float> cpp_proc_action(n_steps * ACT_DIM);

    // ---- 7. 逐帧回放 ----
    std::cout << "[6] 开始逐帧回放..." << std::endl;

    for (size_t t = 0; t < n_steps; ++t) {
        // ===== A: 注入原始传感器数据 =====
        robot->data.root_quat_w = Eigen::Quaternionf(
            raw_imu_quat[t*4 + 0], raw_imu_quat[t*4 + 1],
            raw_imu_quat[t*4 + 2], raw_imu_quat[t*4 + 3]);
        robot->data.root_ang_vel_b = Eigen::Vector3f(
            raw_imu_gyro[t*3 + 0], raw_imu_gyro[t*3 + 1], raw_imu_gyro[t*3 + 2]);
        for (int j = 0; j < NUM_JOINTS; ++j) {
            robot->data.joint_pos[j] = raw_motor_q[t * NUM_JOINTS + j];
            robot->data.joint_vel[j] = raw_motor_dq[t * NUM_JOINTS + j];
        }
        robot->data.waist_yaw = raw_waist_yaw[t];
        robot->data.waist_yaw_omega = raw_waist_yaw_omega[t];

        env.episode_length += 1;
        robot->update();

        // ===== B→C: C++ 独立计算观测 =====
        // 1. 将 motion 帧对齐 Python (motion_window_pos[t] 由 golden trace 保存,
        //    是 Python env step() 中 command_manager._update_command() 输出的 time_steps)
        replay_motion->set_frame(motion_window_pos[t]);

        // 2. 调用 observation_manager.compute() 让 C++ 计算 144 维观测
        //    注意: YAML 无 group 嵌套时, 默认 group 名称为 "obs" (见 _prapare_terms())
        auto cpp_actor_obs = env.observation_manager->compute_group("obs");

        // 3. 拆分为 6 个 term 并保存到 buffer (偏移量与 compare 脚本 OBS_SEGMENTS 一致)
        {
            const float* data = cpp_actor_obs.data();
            std::copy(data + 0,  data + 54,  cpp_obs_motion_command.begin() + t * 54);
            std::copy(data + 54, data + 60,  cpp_obs_motion_anchor_ori_b.begin() + t * 6);
            std::copy(data + 60, data + 63,  cpp_obs_base_ang_vel.begin() + t * 3);
            std::copy(data + 63, data + 90,  cpp_obs_joint_pos.begin() + t * 27);
            std::copy(data + 90, data + 117, cpp_obs_joint_vel.begin() + t * 27);
            std::copy(data + 117, data + 144, cpp_obs_last_action.begin() + t * 27);
        }

        // 4. 构造 ONNX 输入并使用 C++ 独立计算的观测进行推理
        std::unordered_map<std::string, std::vector<float>> obs_map;
        obs_map["obs"] = std::move(cpp_actor_obs);
        auto action = env.alg->act(obs_map);
        env.last_raw_action = action;
        env.action_manager->process_action(action);

        // ===== D: ONNX 推理结果 =====
        std::copy(action.begin(), action.end(),
                  cpp_nn_output.begin() + t * ACT_DIM);

        // ===== F: 处理后的动作 =====
        auto proc = env.action_manager->processed_actions();
        std::copy(proc.begin(), proc.end(),
                  cpp_proc_action.begin() + t * ACT_DIM);

        // ===== 开环注入: 将 Python 的 nn_output 注入为下一帧的 last_action =====
        std::vector<float> py_action(nn_output_py.begin() + t * ACT_DIM,
                                     nn_output_py.begin() + (t + 1) * ACT_DIM);
        env.action_manager->set_action(py_action);
    }

    // ---- 8. 保存 cpp_trace.npz ----
    // B/C 层: 6 个 C++ 独立计算的观测 term (每个通配 cnpy::npz_save 追加写入)
    // 注意: "w" 模式 (create+truncate) 仅在第一个 key 使用, 后续 key 使用 "a" (append)
    std::cout << "[7] 保存 cpp_trace.npz" << std::endl;
    cnpy::npz_save(out_dir + "/cpp_trace.npz", "motion_command",
                   cpp_obs_motion_command.data(), {n_steps, 54}, "w");
    cnpy::npz_save(out_dir + "/cpp_trace.npz", "motion_anchor_ori_b",
                   cpp_obs_motion_anchor_ori_b.data(), {n_steps, 6}, "a");
    cnpy::npz_save(out_dir + "/cpp_trace.npz", "base_ang_vel_pelvis",
                   cpp_obs_base_ang_vel.data(), {n_steps, 3}, "a");
    cnpy::npz_save(out_dir + "/cpp_trace.npz", "joint_pos_rel",
                   cpp_obs_joint_pos.data(), {n_steps, 27}, "a");
    cnpy::npz_save(out_dir + "/cpp_trace.npz", "joint_vel_rel",
                   cpp_obs_joint_vel.data(), {n_steps, 27}, "a");
    cnpy::npz_save(out_dir + "/cpp_trace.npz", "last_action",
                   cpp_obs_last_action.data(), {n_steps, 27}, "a");
    // D/F 层
    cnpy::npz_save(out_dir + "/cpp_trace.npz", "nn_output",
                   cpp_nn_output.data(), {n_steps, (size_t)ACT_DIM}, "a");
    cnpy::npz_save(out_dir + "/cpp_trace.npz", "processed_action",
                   cpp_proc_action.data(), {n_steps, (size_t)ACT_DIM}, "a");

    std::cout << "[OK] C++ trace 保存至 " << out_dir << "/cpp_trace.npz" << std::endl;
    std::cout << "     步数: " << n_steps << std::endl;

    return 0;
}
