#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_RLBase.h"
#include "FSM/State_LocoVelocity.h"
#include "State_Mimic.h"

#include <sstream>

std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
std::shared_ptr<Keyboard> FSMState::keyboard = nullptr;

void init_fsm_state()
{
    auto lowcmd_sub = std::make_shared<unitree::robot::g1::subscription::LowCmd>();
    usleep(0.2 * 1e6);
    if(!lowcmd_sub->isTimeout())
    {
        spdlog::critical("The other process is using the lowcmd channel, please close it first.");
        unitree::robot::go2::shutdown();
        // exit(0);
    }
    FSMState::lowcmd = std::make_unique<LowCmd_t>();
    FSMState::lowstate = std::make_shared<LowState_t>();
    spdlog::info("Waiting for connection to robot...");
    FSMState::lowstate->wait_for_connection();
    spdlog::info("Connected to robot.");
}

int main(int argc, char** argv)
{
    // Load parameters
    auto vm = param::helper(argc, argv);

    std::cout << " --- Unitree Robotics --- \n";
    std::cout << "     H1-2 Controller \n";

    // Unitree DDS Config
    unitree::robot::ChannelFactory::Instance()->Init(0, vm["network"].as<std::string>());

    init_fsm_state();

    FSMState::lowcmd->msg_.mode_machine() = 6;
    if(!FSMState::lowcmd->check_mode_machine(FSMState::lowstate)) {
        spdlog::critical("Unmatched robot type.");
        exit(-1);
    }
    
    // Initialize FSM
    auto fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);

    // Parse --timed schedule if provided
    if (vm.count("timed"))
    {
        std::string timed_str = vm["timed"].as<std::string>();
        std::vector<CtrlFSM::TimedScheduleEntry> schedule;

        std::istringstream ss(timed_str);
        std::string token;
        while (std::getline(ss, token, ','))
        {
            auto colon_pos = token.find(':');
            if (colon_pos == std::string::npos || colon_pos == 0 || colon_pos == token.size() - 1)
            {
                spdlog::warn("Invalid timed schedule entry: '{}' (expected State:seconds)", token);
                continue;
            }
            std::string state_name = token.substr(0, colon_pos);
            try {
                double duration = std::stod(token.substr(colon_pos + 1));
                schedule.push_back({state_name, duration});
            } catch (const std::exception& e) {
                spdlog::warn("Invalid duration in timed schedule entry: '{}'", token);
            }
        }

        if (!schedule.empty())
        {
            spdlog::info("Timed schedule loaded with {} state(s):", schedule.size());
            for (const auto& entry : schedule) {
                spdlog::info("  {} -> {:.1f}s", entry.state_name, entry.duration);
            }
            fsm->setTimedSchedule(schedule);
        }
    }

    fsm->start();

    std::cout << "Press [L2 + Up] to enter FixStand mode.\n";
    std::cout << "From FixStand, press [RB + X] for LocoVelocity (joystick-driven).\n";
    std::cout << "From FixStand, press [RB + Y] for WalkVelocity (RLBase, joystick-driven).\n";
    std::cout << "From FixStand, press [RB + A] for Mimic_Dance1.\n";
    std::cout << "Or use --timed \"FixStand:3,WalkVelocity:10\" for auto-switching.\n";

    while (true)
    {
        sleep(1);
    }
    
    return 0;
}
