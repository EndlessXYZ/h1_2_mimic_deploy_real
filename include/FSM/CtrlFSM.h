// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <unitree/common/thread/recurrent_thread.hpp>
#include "BaseState.h"
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

class CtrlFSM
{
public:
    struct TimedScheduleEntry {
        std::string state_name;
        double duration; // seconds
    };

    CtrlFSM(std::shared_ptr<BaseState> initstate)
    {
        // Initialize FSM states
        states.push_back(std::move(initstate));

    }

    CtrlFSM(YAML::Node cfg)
    {
        auto fsms = cfg["_"]; // enabled FSMs

        // register FSM string map; used for state transition
        for (auto it = fsms.begin(); it != fsms.end(); ++it)
        {
            std::string fsm_name = it->first.as<std::string>();
            int id = it->second["id"].as<int>();
            FSMStringMap.insert({id, fsm_name});
        }

        // Initialize FSM states
        for (auto it = fsms.begin(); it != fsms.end(); ++it)
        {
            std::string fsm_name = it->first.as<std::string>();
            int id = it->second["id"].as<int>();
            std::string fsm_type = it->second["type"] ? it->second["type"].as<std::string>() : fsm_name;
            auto fsm_class = getFsmMap().find("State_" + fsm_type);
            if (fsm_class == getFsmMap().end()) {
                throw std::runtime_error("FSM: Unknown FSM type " + fsm_type);
            }
            auto state_instance = fsm_class->second(id, fsm_name);
            add(state_instance);
        }
    }

    void start() 
    {
        // Reset timed schedule counters
        timed_schedule_index_ = 0;
        elapsed_time_in_state_ = 0.0;

        // Start From State_Passive
        currentState = states[0];
        currentState->enter();

        // If timed schedule starts with a different state, transition immediately
        if (timed_mode_ && timed_schedule_index_ >= 0 && 
            timed_schedule_index_ < (int)timed_schedule_.size())
        {
            auto& first_entry = timed_schedule_[timed_schedule_index_];
            int target_id = fsm_id_from_name(first_entry.state_name);
            if (target_id >= 0 && !currentState->isState(target_id))
            {
                for (auto& state : states)
                {
                    if (state->isState(target_id))
                    {
                        currentState->exit();
                        currentState = state;
                        currentState->enter();
                        elapsed_time_in_state_ = 0.0;
                        spdlog::info("FSM (Timed): Start state {} for {:.1f}s", 
                                     first_entry.state_name, first_entry.duration);
                        break;
                    }
                }
            }
        }

        fsm_thread_ = std::make_shared<unitree::common::RecurrentThread>(
            "FSM", 0, this->dt * 1e6, &CtrlFSM::run_, this);
        spdlog::info("FSM: Start {}", currentState->getStateString());
    }

    void add(std::shared_ptr<BaseState> state)
    {
        for(auto & s : states)
        {
            if(s->isState(state->getState()))
            {
                spdlog::error("FSM: State_{} already exists", state->getStateString());
                std::exit(0);
            }
        }

        states.push_back(std::move(state));
    }

    void setTimedSchedule(const std::vector<TimedScheduleEntry>& schedule)
    {
        timed_schedule_ = schedule;
        timed_mode_ = !schedule.empty();
        timed_schedule_index_ = 0;
        elapsed_time_in_state_ = 0.0;
    }
    
    ~CtrlFSM()
    {
        states.clear();
    }

    std::vector<std::shared_ptr<BaseState>> states;
private:
    const double dt = 0.001;

    void run_()
    {
        currentState->pre_run();
        currentState->run();
        currentState->post_run();
        
        // Check if need to change state (from registered checks)
        int nextStateMode = 0;
        for(int i(0); i<currentState->registered_checks.size(); i++)
        {
            if(currentState->registered_checks[i].first())
            {
                nextStateMode = currentState->registered_checks[i].second;
                break;
            }
        }

        // Timed schedule check (overrides manual transitions)
        if (timed_mode_)
        {
            elapsed_time_in_state_ += dt;

            if (timed_schedule_index_ >= 0 && 
                timed_schedule_index_ < (int)timed_schedule_.size() &&
                elapsed_time_in_state_ >= timed_schedule_[timed_schedule_index_].duration)
            {
                // Advance to next entry in schedule
                timed_schedule_index_++;

                if (timed_schedule_index_ < (int)timed_schedule_.size())
                {
                    auto& entry = timed_schedule_[timed_schedule_index_];
                    int target_id = fsm_id_from_name(entry.state_name);
                    if (target_id >= 0 && !currentState->isState(target_id))
                    {
                        for (auto& state : states)
                        {
                            if (state->isState(target_id))
                            {
                                spdlog::info("FSM (Timed): Change state from {} to {} for {:.1f}s", 
                                             currentState->getStateString(), 
                                             entry.state_name, entry.duration);
                                currentState->exit();
                                currentState = state;
                                currentState->enter();
                                elapsed_time_in_state_ = 0.0;
                                spdlog::info("Timed: entered {}", entry.state_name);
                                break;
                            }
                        }
                    }
                    else if (target_id >= 0)
                    {
                        // Same state, just reset timer
                        elapsed_time_in_state_ = 0.0;
                        spdlog::info("FSM (Timed): Stay in {} for {:.1f}s", 
                                     entry.state_name, entry.duration);
                    }
                }
                else
                {
                    // Schedule complete
                    spdlog::info("FSM (Timed): Schedule completed, staying in {}", 
                                 currentState->getStateString());
                    timed_mode_ = false;
                }
                // Timed transition handled, skip regular checks
                return;
            }
        }

        // Only apply registered checks if not in timed mode (or no timed transition needed)
        if (timed_mode_)
        {
            // In timed mode, still allow safety transitions (e.g. bad_orientation)
            // But only if it's an emergency (check if the transition target is Passive)
            if (nextStateMode != 0 && !currentState->isState(nextStateMode))
            {
                for (auto& state : states)
                {
                    if (state->isState(nextStateMode))
                    {
                        spdlog::warn("FSM (Timed): Safety transition from {} to {}", 
                                     currentState->getStateString(), state->getStateString());
                        currentState->exit();
                        currentState = state;
                        currentState->enter();
                        elapsed_time_in_state_ = 0.0;
                        break;
                    }
                }
            }
        }
        else
        {
            // Normal mode - apply all registered checks
            if (nextStateMode != 0 && !currentState->isState(nextStateMode))
            {
                for (auto& state : states)
                {
                    if (state->isState(nextStateMode))
                    {
                        spdlog::info("FSM: Change state from {} to {}", 
                                     currentState->getStateString(), state->getStateString());
                        currentState->exit();
                        currentState = state;
                        currentState->enter();
                        break;
                    }
                }
            }
        }
    }

    std::shared_ptr<BaseState> currentState;
    unitree::common::RecurrentThreadPtr fsm_thread_;

    // Timed schedule
    std::vector<TimedScheduleEntry> timed_schedule_;
    int timed_schedule_index_ = -1;
    double elapsed_time_in_state_ = 0.0;
    bool timed_mode_ = false;
};
