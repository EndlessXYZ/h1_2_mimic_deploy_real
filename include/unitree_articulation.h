// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "isaaclab/assets/articulation/articulation.h"

namespace unitree
{

// H1_2: waist/torso yaw joint index in the motor_state array (0-indexed).
// Defined once here; all FSM states (Mimic, RLBase, WalkVelocity) and
// articulation update() should reference this constant instead of raw 12.
inline constexpr int TORSO_JOINT_IDX = 12;

template <typename LowStatePtr>
class BaseArticulation : public isaaclab::Articulation
{
public:
    BaseArticulation(LowStatePtr lowstate_)
    : lowstate(lowstate_)
    {
        data.joystick = &lowstate->joystick;
    }

    void update() override
    {
        std::lock_guard<std::mutex> lock(lowstate->mutex_);
        // base_angular_velocity
        for(int i(0); i<3; i++) {
            data.root_ang_vel_b[i] = lowstate->msg_.imu_state().gyroscope()[i];
        }
        // project_gravity_body
        data.root_quat_w = Eigen::Quaternionf(
            lowstate->msg_.imu_state().quaternion()[0],
            lowstate->msg_.imu_state().quaternion()[1],
            lowstate->msg_.imu_state().quaternion()[2],
            lowstate->msg_.imu_state().quaternion()[3]
        );
        // NOTE: projected_gravity_b is in TORSO frame (IMU is mounted on torso_link),
        // NOT pelvis frame. Transform to pelvis via Rz(waist_yaw) if pelvis frame is needed.
        // See observations.h::projected_gravity_pelvis for the correct pelvis-frame variant.
        data.projected_gravity_b = data.root_quat_w.conjugate() * data.GRAVITY_VEC_W;
        // waist yaw (for torso IMU -> pelvis transform)
        data.waist_yaw = lowstate->msg_.motor_state()[TORSO_JOINT_IDX].q();
        data.waist_yaw_omega = lowstate->msg_.motor_state()[TORSO_JOINT_IDX].dq();
        // joint positions and velocities
        for(int i(0); i< data.joint_ids_map.size(); i++) {
            data.joint_pos[i] = lowstate->msg_.motor_state()[data.joint_ids_map[i]].q();
            data.joint_vel[i] = lowstate->msg_.motor_state()[data.joint_ids_map[i]].dq();
        }
    }

    LowStatePtr lowstate;
};

}