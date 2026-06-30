// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "isaaclab/envs/manager_based_rl_env.h"
#include <Eigen/Geometry>

namespace isaaclab
{
namespace mdp
{

REGISTER_OBSERVATION(base_ang_vel)
{
    auto & asset = env->robot;
    auto & data = asset->data.root_ang_vel_b;
    return std::vector<float>(data.data(), data.data() + data.size());
}

// H1_2 特殊处理：IMU 在 torso 上，需要变换到 pelvis frame
// 参考 h1_2_walk_deploy_real/common/rotation_helper.py
REGISTER_OBSERVATION(base_ang_vel_pelvis)
{
    auto & asset = env->robot;
    // waist_yaw 由 BaseArticulation::update() 从 motor_state[12] 读取
    float waist_yaw = asset->data.waist_yaw;
    float waist_yaw_omega = asset->data.waist_yaw_omega;

    // IMU angular velocity (torso frame)
    auto & torso_omega = asset->data.root_ang_vel_b;

    // torso -> pelvis 变换
    Eigen::AngleAxisf waist_rotation(waist_yaw, Eigen::Vector3f::UnitZ());
    Eigen::Vector3f torso_omega_vec(torso_omega[0], torso_omega[1], torso_omega[2]);
    Eigen::Vector3f pelvis_omega = waist_rotation.toRotationMatrix() * torso_omega_vec;
    pelvis_omega[2] -= waist_yaw_omega;

    return std::vector<float>(pelvis_omega.data(), pelvis_omega.data() + 3);
}

// Returns gravity in TORSO frame (IMU on torso_link).
// MJLab H1_2 velocity training uses pelvis frame for projected_gravity
// (root_link = pelvis freejoint body). Use projected_gravity_pelvis for
// the correct pelvis-frame variant to match the training side.
REGISTER_OBSERVATION(projected_gravity)
{
    auto & asset = env->robot;
    auto & data = asset->data.projected_gravity_b;
    return std::vector<float>(data.data(), data.data() + data.size());
}

// H1_2 特殊处理：IMU 在 torso 上，需要变换到 pelvis frame
// 参考 h1_2_walk_deploy_real/common/rotation_helper.py
REGISTER_OBSERVATION(projected_gravity_pelvis)
{
    auto & asset = env->robot;
    // waist_yaw 由 BaseArticulation::update() 从 motor_state[12] 读取
    float waist_yaw = asset->data.waist_yaw;

    // IMU quaternion (torso frame) - w, x, y, z
    auto & torso_quat = asset->data.root_quat_w;

    // torso -> pelvis 变换：R_pelvis = R_torso * RzWaist^T
    Eigen::AngleAxisf waist_rotation(waist_yaw, Eigen::Vector3f::UnitZ());
    Eigen::Quaternionf pelvis_quat = torso_quat * waist_rotation.inverse();

    // 投影重力向量
    Eigen::Vector3f gravity_world(0.0f, 0.0f, -1.0f);
    Eigen::Vector3f gravity_pelvis = pelvis_quat.conjugate() * gravity_world;

    return std::vector<float>(gravity_pelvis.data(), gravity_pelvis.data() + 3);
}

REGISTER_OBSERVATION(joint_pos)
{
    auto & asset = env->robot;
    std::vector<float> data;

    std::vector<int> joint_ids;
    try {
        joint_ids = params["asset_cfg"]["joint_ids"].as<std::vector<int>>();
    } catch(const std::exception& e) {
    }

    if(joint_ids.empty())
    {
        data.resize(asset->data.joint_pos.size());
        for(size_t i = 0; i < asset->data.joint_pos.size(); ++i)
        {
            data[i] = asset->data.joint_pos[i];
        }
    }
    else
    {
        data.resize(joint_ids.size());
        for(size_t i = 0; i < joint_ids.size(); ++i)
        {
            data[i] = asset->data.joint_pos[joint_ids[i]];
        }
    }

    return data;
}

REGISTER_OBSERVATION(joint_pos_rel)
{
    auto & asset = env->robot;
    std::vector<float> data;

    data.resize(asset->data.joint_pos.size());
    for(size_t i = 0; i < asset->data.joint_pos.size(); ++i) {
        data[i] = asset->data.joint_pos[i] - asset->data.default_joint_pos[i];
    }

    try {
        std::vector<int> joint_ids;
        joint_ids = params["asset_cfg"]["joint_ids"].as<std::vector<int>>();
        if(!joint_ids.empty()) {
            std::vector<float> tmp_data;
            tmp_data.resize(joint_ids.size());
            for(size_t i = 0; i < joint_ids.size(); ++i){
                tmp_data[i] = data[joint_ids[i]];
            }
            data = tmp_data;
        }
    } catch(const std::exception& e) {
    
    }

    return data;
}

REGISTER_OBSERVATION(joint_vel_rel)
{
    auto & asset = env->robot;
    std::vector<float> data;

    // Subtract default_joint_vel to match Python training side
    data.resize(asset->data.joint_vel.size());
    for(size_t i = 0; i < asset->data.joint_vel.size(); ++i) {
        data[i] = asset->data.joint_vel[i] - asset->data.default_joint_vel[i];
    }

    try {
        const std::vector<int> joint_ids = params["asset_cfg"]["joint_ids"].as<std::vector<int>>();

        if(!joint_ids.empty()) {
            std::vector<float> filtered;
            filtered.resize(joint_ids.size());
            for(size_t i = 0; i < joint_ids.size(); ++i) {
                filtered[i] = data[joint_ids[i]];
            }
            data = filtered;
        }
    } catch(const std::exception& e) {
    }
    return data;
}

REGISTER_OBSERVATION(last_action)
{
    auto data = env->action_manager->action();
    return std::vector<float>(data.data(), data.data() + data.size());
};

// 速度指令观测
// 训练侧 command 范围由 velocity_command.py 定义（play 模式 ranges），
// 部署侧将 joystick [-1, 1] 直接映射到该范围：vx = joy * max_vx 等。
REGISTER_OBSERVATION(velocity_commands)
{
    auto & asset = env->robot;
    auto joystick = asset->data.joystick;

    if (!joystick) {
        return std::vector<float>{0.0f, 0.0f, 0.0f};
    }

    // 从 params 中读取 ranges 配置
    float lin_vel_x_max = 1.0f;
    float lin_vel_y_max = 0.5f;
    float ang_vel_z_max = 0.5f;

    if (params["ranges"]) {
        auto ranges = params["ranges"];
        if (ranges["lin_vel_x"]) {
            lin_vel_x_max = ranges["lin_vel_x"]["upper"].as<float>();
        }
        if (ranges["lin_vel_y"]) {
            lin_vel_y_max = ranges["lin_vel_y"]["upper"].as<float>();
        }
        if (ranges["ang_vel_z"]) {
            ang_vel_z_max = ranges["ang_vel_z"]["upper"].as<float>();
        }
    }

    // joystick 输入范围 [-1, 1]，直接映射到 command 范围
    float vx = joystick->ly() * lin_vel_x_max;
    float vy = -joystick->lx() * lin_vel_y_max;
    float wz = -joystick->rx() * ang_vel_z_max;

    return std::vector<float>{vx, vy, wz};
}

REGISTER_OBSERVATION(gait_phase)
{
    float period = params["period"].as<float>();
    float delta_phase = env->step_dt * (1.0f / period);

    env->global_phase += delta_phase;
    env->global_phase = std::fmod(env->global_phase, 1.0f);

    std::vector<float> obs(2);
    obs[0] = std::sin(env->global_phase * 2 * M_PI);
    obs[1] = std::cos(env->global_phase * 2 * M_PI);

    return obs;
}

}
}