#pragma once

#include "isaaclab/envs/manager_based_rl_env.h"

namespace isaaclab
{
namespace mdp
{

// FSM thread variant: compute bad_orientation from raw lowstate IMU/motor data directly.
// The FSM thread calls lowstate->update() in pre_run(), so IMU and waist_yaw data
// are fresh (updated at 1kHz). This decouples fall detection from the policy thread,
// ensuring falls are detected even if the policy thread stalls.
inline bool bad_orientation(const Eigen::Quaternionf& imu_quat, float waist_yaw, float limit_angle = 1.0f)
{
    // Compute projected gravity in IMU (torso) frame: g_b = q_conj * g_w
    Eigen::Vector3f gravity_world(0.0f, 0.0f, -1.0f);
    Eigen::Vector3f projected_gravity_b = imu_quat.conjugate() * gravity_world;

    // Transform from torso frame to pelvis frame using waist_yaw rotation
    Eigen::AngleAxisf waist_rotation(waist_yaw, Eigen::Vector3f::UnitZ());
    Eigen::Vector3f gravity_pelvis = waist_rotation.toRotationMatrix() * projected_gravity_b;

    // Clamp to [-1, 1] to avoid NaN from acos due to float32 imprecision.
    float cos_theta = std::clamp(-gravity_pelvis[2], -1.0f, 1.0f);
    return std::acos(cos_theta) > limit_angle;
}

} 
} 