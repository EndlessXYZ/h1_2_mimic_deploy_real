// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSMState.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"

class State_RLBase : public FSMState
{
public:
    State_RLBase(int state_mode, std::string state_string);
    
    void enter()
    {
        // set gain
        for (int i = 0; i < env->robot->data.joint_stiffness.size(); ++i)
        {
            lowcmd->msg_.motor_cmd()[i].kp() = env->robot->data.joint_stiffness[i];
            lowcmd->msg_.motor_cmd()[i].kd() = env->robot->data.joint_damping[i];
            lowcmd->msg_.motor_cmd()[i].dq() = 0;
            lowcmd->msg_.motor_cmd()[i].tau() = 0;
        }

        env->robot->update();
        // Start policy thread
        policy_thread_running = true;
        policy_thread = std::thread([this]{
            using clock = std::chrono::high_resolution_clock;
            const std::chrono::duration<double> desiredDuration(env->step_dt);
            const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

            // Initialize timing
            auto sleepTill = clock::now() + dt;

            // Align default_joint_pos to current robot pose so joint_pos_rel ≈ 0.
            // Critical for smooth transition from an arbitrary pose (e.g. after Mimic).
            env->robot->data.default_joint_pos = env->robot->data.joint_pos;
            env->reset();

            // Initialize last_action from current joint positions to avoid
            // observation contradiction on the first policy step.
            // Without this, last_action would be all zeros while joint_pos_rel
            // is near zero (from reset), causing the policy to output extreme values.
            auto action_scale = env->cfg["actions"]["JointPositionAction"]["scale"].as<std::vector<float>>();
            auto action_offset = env->cfg["actions"]["JointPositionAction"]["offset"].as<std::vector<float>>();
            std::vector<float> initial_action(env->robot->data.joint_pos.size());
            for(size_t i = 0; i < env->robot->data.joint_pos.size(); ++i) {
                initial_action[i] = (env->robot->data.joint_pos[i] - action_offset[i]) / action_scale[i];
            }
            env->action_manager->set_action(initial_action);
            env->observation_manager->recompute_term_history("last_action");
            env->action_manager->process_action(initial_action);

            while (policy_thread_running)
            {
                env->step();

                // Sleep
                std::this_thread::sleep_until(sleepTill);
                sleepTill += dt;
            }
        });
    }

    void run();
    
    void exit()
    {
        policy_thread_running = false;
        if (policy_thread.joinable()) {
            policy_thread.join();
        }
    }

private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;

    std::thread policy_thread;
    bool policy_thread_running = false;
};

REGISTER_FSM(State_RLBase)
