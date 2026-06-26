// SKILL Step 3: C++ 侧离线回放验证器
//
// 读取 Python 采集的 golden_trace.npz，注入原始传感器数据到真实的
// ManagerBasedRLEnv + ObservationManager + ActionManager + OrtRunner 管线，
// 逐帧跑 A→F 全链路，dump 每层输出到 cpp_trace.npz。
//
// 用法: ./replay_validator <golden_trace.npz> <deploy.yaml> <policy.onnx> <output_dir>

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
#include <unitree/dds_wrapper/common/unitree_joystick.hpp>

using namespace isaaclab;
using unitree::common::UnitreeJoystick;
using unitree::common::Axis;

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
// 辅助函数：直接设置 Axis 的内部 data_ 值（绕过平滑滤波）
// ============================================================================
static void set_axis_value(Axis& axis, float value) {
    // Axis::operator()() 返回 const float& → 指向私有 data_
    // 通过 const_cast 直接修改 data_，绕过 operator()(float) 的平滑逻辑
    const float& ref = axis();
    const_cast<float&>(ref) = value;
}

// ============================================================================
// 辅助函数：从 cnpy::NpyArray 读取 float 数据
// ============================================================================
static std::vector<float> read_float_vec(const cnpy::NpyArray& arr) {
    if (arr.word_size == sizeof(float)) {
        return arr.as_vec<float>();
    }
    // 如果是 double，转换为 float
    auto d = arr.as_vec<double>();
    std::vector<float> f(d.size());
    for (size_t i = 0; i < d.size(); ++i) f[i] = static_cast<float>(d[i]);
    return f;
}

// ============================================================================
// main
// ============================================================================
int main(int argc, char* argv[])
{
    if (argc < 5) {
        std::cerr << "用法: " << argv[0]
                  << " <golden_trace.npz> <deploy.yaml> <policy.onnx> <output_dir>\n";
        return 1;
    }

    std::string golden_path = argv[1];
    std::string yaml_path   = argv[2];
    std::string onnx_path   = argv[3];
    std::string out_dir     = argv[4];

    // ---- 1. 加载 golden trace ----
    std::cout << "[1] 加载 golden trace: " << golden_path << std::endl;
    auto npz = cnpy::npz_load(golden_path);

    auto raw_imu_quat  = read_float_vec(npz["raw_imu_quat"]);
    auto raw_imu_gyro  = read_float_vec(npz["raw_imu_gyro"]);
    auto raw_motor_q   = read_float_vec(npz["raw_motor_q"]);
    auto raw_motor_dq  = read_float_vec(npz["raw_motor_dq"]);
    auto raw_waist_yaw = read_float_vec(npz["raw_waist_yaw"]);
    auto raw_waist_yaw_omega = read_float_vec(npz["raw_waist_yaw_omega"]);
    auto raw_joystick  = read_float_vec(npz["raw_joystick"]);
    auto obs_scaled_py = read_float_vec(npz["obs_scaled"]);
    auto nn_output_py  = read_float_vec(npz["nn_output"]);
    auto proc_action_py = read_float_vec(npz["processed_action"]);

    size_t n_steps = npz["raw_imu_quat"].shape[0];
    std::cout << "    步数: " << n_steps << std::endl;

    // ---- 2. 加载配置 & 创建环境 ----
    std::cout << "[2] 加载配置: " << yaml_path << std::endl;
    YAML::Node cfg = YAML::LoadFile(yaml_path);

    auto robot = std::make_shared<ReplayArticulation>();
    auto joystick = std::make_shared<UnitreeJoystick>();
    robot->data.joystick = joystick.get();

    // 预分配 data 字段
    int n_joints = cfg["joint_ids_map"].as<std::vector<float>>().size();
    robot->data.joint_pos.resize(n_joints);
    robot->data.joint_vel.resize(n_joints);
    robot->data.root_quat_w = Eigen::Quaternionf(1, 0, 0, 0);
    robot->data.root_ang_vel_b = Eigen::Vector3f::Zero();
    robot->data.projected_gravity_b = Eigen::Vector3f(0, 0, -1);
    robot->data.waist_yaw = 0.0f;
    robot->data.waist_yaw_omega = 0.0f;

    ManagerBasedRLEnv env(cfg, robot);
    env.alg = std::make_unique<OrtRunner>(onnx_path);

    // ---- 3. 重置环境 ----
    std::cout << "[3] 重置环境" << std::endl;
    env.action_manager->reset();       // action = 0
    env.observation_manager->reset();  // 使用 action=0 初始化 last_action 历史
    env.global_phase = 0.0f;
    env.episode_length = 0;

    // ---- 4. 分配输出 buffer ----
    const int OBS_DIM = 47;
    const int ACT_DIM = 12;

    std::vector<float> cpp_obs_scaled(n_steps * OBS_DIM);
    std::vector<float> cpp_nn_output(n_steps * ACT_DIM);
    std::vector<float> cpp_proc_action(n_steps * ACT_DIM);

    // ---- 5. 逐帧回放 ----
    std::cout << "[4] 开始逐帧回放..." << std::endl;

    for (size_t t = 0; t < n_steps; ++t) {
        // ===== A: 注入原始传感器数据 =====
        // IMU 四元数 (w, x, y, z)
        robot->data.root_quat_w = Eigen::Quaternionf(
            raw_imu_quat[t*4 + 0],
            raw_imu_quat[t*4 + 1],
            raw_imu_quat[t*4 + 2],
            raw_imu_quat[t*4 + 3]
        );
        // IMU 角速度 (torso frame)
        robot->data.root_ang_vel_b = Eigen::Vector3f(
            raw_imu_gyro[t*3 + 0],
            raw_imu_gyro[t*3 + 1],
            raw_imu_gyro[t*3 + 2]
        );
        // 电机位置
        for (int j = 0; j < n_joints; ++j)
            robot->data.joint_pos[j] = raw_motor_q[t * n_joints + j];
        // 电机速度
        for (int j = 0; j < n_joints; ++j)
            robot->data.joint_vel[j] = raw_motor_dq[t * n_joints + j];
        // waist_yaw (独立存储，不在 raw_motor_q 中)
        robot->data.waist_yaw = raw_waist_yaw[t];
        robot->data.waist_yaw_omega = raw_waist_yaw_omega[t];

        // 摇杆 (直接设置 Axis 内部值，绕过平滑)
        float cmd_x = raw_joystick[t * 3 + 0];
        float cmd_y = raw_joystick[t * 3 + 1];
        float cmd_z = raw_joystick[t * 3 + 2];
        set_axis_value(joystick->ly, cmd_x);
        set_axis_value(joystick->lx, -cmd_y);  // velocity_commands 里 lx 取负
        set_axis_value(joystick->rx, -cmd_z);  // velocity_commands 里 rx 取负

        // ===== B~C: 观测计算 (通过 ObservationManager) =====
        env.step();

        // 提取 C: scaled obs (env->last_observation 是 compute() 的拼接结果)
        std::copy(env.last_observation.begin(), env.last_observation.end(),
                  cpp_obs_scaled.begin() + t * OBS_DIM);

        // ===== D: ONNX 推理结果 =====
        std::copy(env.last_raw_action.begin(), env.last_raw_action.end(),
                  cpp_nn_output.begin() + t * ACT_DIM);

        // ===== F: 处理后的动作 =====
        auto proc = env.action_manager->processed_actions();
        std::copy(proc.begin(), proc.end(),
                  cpp_proc_action.begin() + t * ACT_DIM);

        // ===== 开环注入: 将 Python 的 raw action 注入为 C++ 的 last_action =====
        // 这样下一步的 last_action 观测与 Python 完全一致，避免误差累积
        std::vector<float> py_action(nn_output_py.begin() + t * ACT_DIM,
                                     nn_output_py.begin() + (t + 1) * ACT_DIM);
        env.action_manager->set_action(py_action);
    }

    // ---- 6. 保存 cpp_trace.npz ----
    std::cout << "[5] 保存 cpp_trace.npz" << std::endl;
    cnpy::npz_save(out_dir + "/cpp_trace.npz", "obs_scaled",
                   cpp_obs_scaled.data(), {n_steps, (size_t)OBS_DIM}, "w");
    cnpy::npz_save(out_dir + "/cpp_trace.npz", "nn_output",
                   cpp_nn_output.data(), {n_steps, (size_t)ACT_DIM}, "a");
    cnpy::npz_save(out_dir + "/cpp_trace.npz", "processed_action",
                   cpp_proc_action.data(), {n_steps, (size_t)ACT_DIM}, "a");

    std::cout << "[OK] C++ trace 保存至 " << out_dir << "/cpp_trace.npz" << std::endl;
    std::cout << "     步数: " << n_steps << std::endl;

    return 0;
}
