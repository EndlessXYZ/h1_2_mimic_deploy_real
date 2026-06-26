#pragma once

#include "FSM/State_RLBase.h"
#include <cnpy.h>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <Eigen/Core>
#include <Eigen/Geometry>


class State_Mimic : public FSMState
{
public:
    State_Mimic(int state_mode, std::string state_string);

    void enter();
    void run();
    void exit()
    {
        policy_thread_running = false;
        if (policy_thread.joinable()) {
            policy_thread.join();
        }
        if (dump_file_.is_open()) {
            dump_file_.close();
        }
    }

    class MotionLoader_;

    static std::shared_ptr<MotionLoader_> motion; // for obs computation

    // SKILL 开环注入: 初始化 replay 所需的静态变量 (motion 指针 + init_quat)
    static void init_replay(std::shared_ptr<MotionLoader_> motion_ptr,
                            const Eigen::Quaternionf& ref_quat,
                            const Eigen::Quaternionf& robot_quat);
private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;
    std::shared_ptr<MotionLoader_> motion_; // for saving

    std::thread policy_thread;
    bool policy_thread_running = false;
    std::array<float, 2> time_range_;

    // CSV dump
    std::ofstream dump_file_;
    bool dump_enabled_ = false;
};


class State_Mimic::MotionLoader_
{
public:
    MotionLoader_(std::string motion_file)
    {
        load_data_from_npz(motion_file);
        num_frames = dof_positions.size();
        dt = 1.0f / fps;
        duration = num_frames * dt;

        // Validate loaded motion data for NaN/INF
        for (const auto& pos : dof_positions) {
            for (int i = 0; i < pos.size(); i++) {
                if (std::isnan(pos[i]) || std::isinf(pos[i])) {
                    throw std::runtime_error("Motion file contains NaN/INF in joint_pos");
                }
            }
        }

        update(0.0f);
    }

    void load_data_from_npz(const std::string& motion_file)
    {
        cnpy::npz_t npz_data = cnpy::npz_load(motion_file);

        auto body_pos_w  = npz_data["body_pos_w"];   // [frame, body_id, 3]
        auto body_quat_w = npz_data["body_quat_w"];  // [frame, body_id, 4]
        auto joint_pos   = npz_data["joint_pos"];    // [frame, dof]
        auto joint_vel   = npz_data["joint_vel"];    // [frame, dof]
        auto fps_array  = npz_data["fps"];            // [1]
        // numpy 默认 float64 存储, 但 C++ cnpy data<float>() 只读取底层字节不转换
        // 若 fps 实际是 float64 而按 float32 读取, 会得到 0.0 (IEEE 754 高位字节为 0)
        // 导致 dt = 1/0 = inf, 所有帧索引为 0, MotionLoader_ 始终返回第 0 帧数据
        if (fps_array.word_size == sizeof(float)) {
            fps = fps_array.data<float>()[0];
        } else {
            fps = static_cast<float>(fps_array.data<double>()[0]);
        }

        root_positions.clear();
        root_quaternions.clear();
        dof_positions.clear();
        dof_velocities.clear();

        const size_t num_frames_npz = body_pos_w.shape[0];

        for (size_t i = 0; i < num_frames_npz; i++)
        {
            const size_t body_stride_pos  = body_pos_w.shape[1] * body_pos_w.shape[2];
            const size_t body_stride_quat = body_quat_w.shape[1] * body_quat_w.shape[2];

            Eigen::Vector3f root_pos = Eigen::Vector3f::Map(body_pos_w.data<float>() + i * body_stride_pos);
            root_positions.push_back(root_pos);

            Eigen::Quaternionf quat(
                body_quat_w.data<float>()[i * body_stride_quat + 0], // w
                body_quat_w.data<float>()[i * body_stride_quat + 1], // x
                body_quat_w.data<float>()[i * body_stride_quat + 2], // y
                body_quat_w.data<float>()[i * body_stride_quat + 3]  // z
            );
            root_quaternions.push_back(quat);

            Eigen::VectorXf joint_position(joint_pos.shape[1]);
            for (int j = 0; j < joint_pos.shape[1]; j++) {
                joint_position[j] = joint_pos.data<float>()[i * joint_pos.shape[1] + j];
            }

            Eigen::VectorXf joint_velocity(joint_vel.shape[1]);
            for (int j = 0; j < joint_vel.shape[1]; j++) {
                joint_velocity[j] = joint_vel.data<float>()[i * joint_vel.shape[1] + j];
            }

            dof_positions.push_back(joint_position);
            dof_velocities.push_back(joint_velocity);
        }
    }

    void update(float time)
    {
        float phase = std::fmod(time, duration);
        float f = phase / dt;
        // 加 epsilon 防止 float32 累积精度问题:
        // e.g. N=27 步时: 27 * 0.02f / 0.02f = 26.999... (而非 27.0)
        // floor(26.999) = 26, 比正确帧索引少 1 → 全步错位, motion_anchor_ori_b 误差跳变
        // epsilon=1e-4f: 大于 float32 累积误差上限 (~4e-7), 小于 1 帧间距 (~0.02),
        // 确保 N*dt/dt 在整数附近的浮点误差被正确归整
        frame = static_cast<int>(std::floor(f + 1e-4f));
        frame = std::min(frame, num_frames - 1);
    }

    void reset(const isaaclab::ArticulationData & data, float t = 0.0f)
    {
        update(t);
        auto init_to_anchor = isaaclab::yawQuaternion(this->root_quaternion()).toRotationMatrix();
        auto world_to_anchor = isaaclab::yawQuaternion(data.root_quat_w).toRotationMatrix();
        world_to_init_ = world_to_anchor * init_to_anchor.transpose();
    }

    Eigen::VectorXf root_position() {
        return root_positions[frame];
    }
    Eigen::Quaternionf root_quaternion() {
        return root_quaternions[frame];
    }
    Eigen::VectorXf joint_pos() {
        return dof_positions[frame];
    }
    Eigen::VectorXf joint_vel() {
        return dof_velocities[frame];
    }

    float dt;
    float fps;
    int num_frames;
    float duration;

    int frame;
    std::vector<Eigen::VectorXf> root_positions;
    std::vector<Eigen::Quaternionf> root_quaternions;
    std::vector<Eigen::VectorXf> dof_positions;
    std::vector<Eigen::VectorXf> dof_velocities;
    Eigen::Matrix3f world_to_init_;
};


REGISTER_FSM(State_Mimic)
