// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSMState.h"
#include <unitree/robot/h1/loco/h1_loco_client.hpp>
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>

class State_LocoVelocity : public FSMState
{
public:
    State_LocoVelocity(int state, std::string state_string = "LocoVelocity")
    : FSMState(state, state_string)
    {
        auto cfg = param::config["FSM"][state_string];

        max_vx_    = cfg["max_vx"]    ? cfg["max_vx"].as<float>()    : 0.5f;
        max_vy_    = cfg["max_vy"]    ? cfg["max_vy"].as<float>()    : 0.3f;
        max_omega_ = cfg["max_omega"] ? cfg["max_omega"].as<float>() : 0.5f;
        swing_height_ = cfg["swing_height"] ? cfg["swing_height"].as<float>() : 0.08f;
        stand_height_ = cfg["stand_height"] ? cfg["stand_height"].as<float>() : 0.0f; // 0 = use default
        send_interval_ms_ = cfg["send_interval_ms"] ? cfg["send_interval_ms"].as<int>() : 20;

        spdlog::info("State_LocoVelocity: max_vx={}, max_vy={}, max_omega={}, swing_height={}, stand_height={}",
                     max_vx_, max_vy_, max_omega_, swing_height_, stand_height_);
    }

    void enter()
    {
        spdlog::info("State_LocoVelocity: === Entering (high-level loco control) ===");

        // ====================================================================
        // Phase 0: Mode Switch via MotionSwitcherClient
        // On real robot, low-level FSM (Passive/FixStand) runs with loco_service
        // released. We must activate sport_mode before LocoClient RPC calls
        // can take effect. Otherwise the RPCs will timeout or bounce.
        // ====================================================================
        msc_.SetTimeout(10.0f);
        msc_.Init();

        // 0a. Release any active low-level motion service (idempotent)
        std::string form, motion_name;
        if (msc_.CheckMode(form, motion_name) == 0 && !motion_name.empty()) {
            spdlog::info("MotionSwitcher: Active service '{}' detected, releasing...", motion_name);
            for (int i = 0; i < 5; ++i) {
                int32_t ret = msc_.ReleaseMode();
                if (ret == 0) {
                    spdlog::info("MotionSwitcher: ReleaseMode OK");
                    break;
                }
                spdlog::warn("MotionSwitcher: ReleaseMode attempt {}/5 failed (ret={})", i+1, ret);
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        } else {
            spdlog::info("MotionSwitcher: No active service to release.");
        }

        // 0b. Activate sport_mode (loco_service)
        spdlog::info("MotionSwitcher: Activating sport_mode...");
        int32_t ret = msc_.SelectMode("normal");
        if (ret != 0) {
            spdlog::error("MotionSwitcher: SelectMode('normal') failed (ret={}). "
                          "Ensure robot is connected and in Damp mode.", ret);
        }
        std::this_thread::sleep_for(std::chrono::seconds(2));

        // ====================================================================
        // Phase 1: LocoClient Init & Damping Safety Island
        // Always start from Damp to eliminate torque step from previous state.
        // This is the "damping safety island" — zero stiffness, safe transition.
        // ====================================================================
        client_.Init();
        client_.SetTimeout(10.f);

        // Safety island: force Damp (Kp=0, low Kd)
        spdlog::info("LocoClient: Entering Damp (safety island)...");
        client_.Damp();
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        // ====================================================================
        // Phase 2: Stand Up
        // ====================================================================
        spdlog::info("LocoClient: Standing up...");
        client_.StandUp();
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        // Two-phase polling to avoid a race where GetFsmId is read before
        // loco_service has processed the StandUp() RPC and is still in Damp (id=1):
        //   Phase A: wait until loco_service ACKs it actually entered StandUp (fsm_id == 2)
        //   Phase B: wait until it autonomously leaves StandUp (fsm_id != 2)
        // The previous single-loop check (fsm_id != 2) fired immediately because
        // the default fsm_id was 0 and the RPC had not yet taken effect, causing
        // Start() to be issued prematurely.
        int fsm_id = 0;
        bool entered_standup = false;

        // Phase A: confirm StandUp (fsm_id == 2) is reached
        for (int i = 0; i < 25; ++i) {
            if (client_.GetFsmId(fsm_id) == 0 && fsm_id == 2) {
                entered_standup = true;
                spdlog::info("LocoClient: loco_service entered StandUp (fsm_id=2)");
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        if (!entered_standup) {
            spdlog::warn("LocoClient: StandUp not confirmed (fsm_id={}). Proceeding cautiously.", fsm_id);
        }

        // Phase B: wait for loco_service to autonomously leave StandUp (fsm_id != 2)
        for (int i = 0; i < 30; ++i) {
            if (client_.GetFsmId(fsm_id) == 0 && fsm_id != 2) {
                spdlog::info("LocoClient: StandUp complete (fsm_id={})", fsm_id);
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }

        // ====================================================================
        // Phase 3: Start Locomotion
        // ====================================================================
        spdlog::info("LocoClient: Starting locomotion mode...");
        client_.Start();
        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        // ====================================================================
        // Phase 4: Configure Gait Parameters
        // ====================================================================
        client_.ContinuousGait(true);  // 原地踏步保持动态平衡
        client_.SetSwingHeight(swing_height_);
        if (stand_height_ > 0.01f) {
            client_.SetStandHeight(stand_height_);
        }
        client_.EnableOdom();

        client_initialized_ = true;
        last_send_time_ = std::chrono::steady_clock::now();

        spdlog::info("State_LocoVelocity: === Ready. Use joystick (ly=forward, lx=strafe, rx=yaw) ===");
    }

    void run()
    {
        if (!client_initialized_) return;

        auto now = std::chrono::steady_clock::now();
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_send_time_).count();

        if (elapsed_ms < send_interval_ms_) return;
        last_send_time_ = now;

        // Read joystick axes. Axis sign follows the same robot-frame
        // convention as observations.h (lin_vel_y = -lx, ang_vel_z = -rx):
        //  - robot x (forward) is positive ly, no flip
        //  - robot y (left)   is positive -lx  → lx pushed right means vy<0
        //  - yaw (CCW)        is positive -rx  → rx pushed right means turn right
        float ly = lowstate->joystick.ly();
        float lx = lowstate->joystick.lx();
        float rx = lowstate->joystick.rx();

        float vx    = ly * max_vx_;
        float vy    = -lx * max_vy_;
        float omega = -rx * max_omega_;

        // Duration: 3x the send interval plus 0.5s buffer for safety
        float duration = std::max(0.5f, send_interval_ms_ * 3.0f / 1000.0f);
        client_.SetVelocity(vx, vy, omega, duration);
    }

    void exit()
    {
        spdlog::info("State_LocoVelocity: === Exiting (handing back to low-level FSM) ===");

        // Use shorter timeout for exit RPCs so that a DDS disconnect
        // doesn't block the 1kHz FSM thread for 10s per call.
        client_.SetTimeout(2.0f);

        // Stop movement and enter Damp — always attempted regardless of
        // client_initialized_, because client_ is Init()'d in enter() and
        // the RPCs are idempotent/safe from any loco_service state.
        // This also handles the case where enter() failed mid-way:
        // sport_mode was activated but Start() never ran.
        client_.StopMove();
        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        client_.DisableOdom();

        // Return to Damp via loco_service (safety island before mode switch)
        client_.Damp();
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        client_initialized_ = false;

        // Release sport_mode — returns bus to low-level control
        msc_.SetTimeout(2.0f);
        spdlog::info("MotionSwitcher: Releasing sport_mode...");
        for (int i = 0; i < 3; ++i) {
            int32_t ret = msc_.ReleaseMode();
            if (ret == 0) {
                spdlog::info("MotionSwitcher: ReleaseMode OK");
                break;
            }
            spdlog::warn("MotionSwitcher: ReleaseMode attempt {}/3 failed (ret={})", i+1, ret);
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        spdlog::info("State_LocoVelocity: === Exited. Lowcmd bus is free for low-level FSM. ===");
    }

    void post_run()
    {
        // 空覆写 — loco_service 内部控制电机，不通过 lowcmd 发布关节指令
        // 如果不禁用此方法，CtrlFSM 的 1kHz 线程将持续发布 FixStand 的关节位姿，
        // 与 loco_service 产生总线争夺，导致电机剧烈抖震或损坏减速器
    }

private:
    unitree::robot::b2::MotionSwitcherClient msc_;
    unitree::robot::h1::LocoClient client_;
    bool client_initialized_ = false;

    float max_vx_   = 0.5f;
    float max_vy_   = 0.3f;
    float max_omega_ = 0.5f;
    float swing_height_ = 0.08f;
    float stand_height_ = 0.0f; // 0 = use default
    int send_interval_ms_ = 20; // 50Hz throttle

    std::chrono::steady_clock::time_point last_send_time_;
};

REGISTER_FSM(State_LocoVelocity)
