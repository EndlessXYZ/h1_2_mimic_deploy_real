#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string) 
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{
                // Compute bad_orientation from raw lowstate data in the FSM thread
                // (pre_run() calls lowstate->update(), so IMU/motor data is fresh).
                // This decouples fall detection from the policy thread's robot->update().
                auto & imu = lowstate->msg_.imu_state();
                Eigen::Quaternionf imu_quat(
                    imu.quaternion()[0], imu.quaternion()[1],
                    imu.quaternion()[2], imu.quaternion()[3]
                );
                float waist_yaw = lowstate->msg_.motor_state()[unitree::TORSO_JOINT_IDX].q();
                return isaaclab::mdp::bad_orientation(imu_quat, waist_yaw, 1.0f);
            },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_RLBase::run()
{
    auto action = env->action_manager->processed_actions();
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}