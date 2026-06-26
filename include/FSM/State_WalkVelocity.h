// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.
// 
// H1_2 Velocity 行走策略状态类
// 基于 unitree_rl_gym/deploy/pre_train/h1_2/motion.onnx
// 观测维度: 47 (仅腿部)
// 动作维度: 12 (仅腿部)

#pragma once

#include "FSM/State_RLBase.h"
#include <cmath>
#include <fstream>
#include <chrono>


class State_WalkVelocity : public FSMState
{
public:
    State_WalkVelocity(int state_mode, std::string state_string);

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

private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;

    std::thread policy_thread;
    bool policy_thread_running = false;

    // 手臂/腰部关节固定目标位置
    std::vector<float> arm_waist_target_;
    std::vector<int> arm_waist_joint_idx_;
    std::vector<float> arm_waist_kp_;
    std::vector<float> arm_waist_kd_;

    // CSV dump
    std::ofstream dump_file_;
    bool dump_enabled_ = false;

    // PD gain smooth transition
    std::chrono::steady_clock::time_point enter_time_;
    bool pd_transition_active_ = false;
    float pd_transition_duration_ = 0.1f; // seconds
    std::vector<float> prev_kp_;  // previous Kp (from FixStand)
    std::vector<float> prev_kd_;  // previous Kd (from FixStand)
};


REGISTER_FSM(State_WalkVelocity)