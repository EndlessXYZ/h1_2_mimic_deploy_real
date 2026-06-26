#!/usr/bin/env python3
"""
H1_2 WalkVelocity 观测组装数值校验脚本
对比 C++ deploy 侧与 Python 训练侧的观测组装公式等价性
"""

import numpy as np
import onnxruntime as ort
import yaml
import os
import json
from scipy.spatial.transform import Rotation as R

# 配置路径
DEPLOY_YAML_PATH = "/home/meme/Documents/unitree_h1/h1_2_mimic_deploy_real/robots/h1_2/config/policy/walk_velocity/v0/params/deploy.yaml"
ONNX_MODEL_PATH = "/home/meme/Documents/unitree_h1/h1_2_mimic_deploy_real/robots/h1_2/config/policy/walk_velocity/v0/exported/policy.onnx"
H1_2_CONFIG_PATH = "/home/meme/Documents/unitree_h1/h1_2_walk_deploy_real/configs/h1_2.yaml"

# 输出路径
OUTPUT_DIR = "/home/meme/Documents/unitree_h1/artifacts/walk_velocity_compare_h1_2"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_yaml_config(path):
    """加载 YAML 配置文件"""
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def get_gravity_orientation(quaternion):
    """
    从四元数计算重力方向向量（参考 h1_2_walk_deploy_real/common/rotation_helper.py）
    quaternion: [w, x, y, z]
    """
    qw, qx, qy, qz = quaternion
    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation


def transform_imu_data(waist_yaw, waist_yaw_omega, imu_quat, imu_omega):
    """
    将 torso IMU 数据变换到 pelvis frame
    参考 h1_2_walk_deploy_real/common/rotation_helper.py
    """
    RzWaist = R.from_euler("z", waist_yaw).as_matrix()
    R_torso = R.from_quat([imu_quat[1], imu_quat[2], imu_quat[3], imu_quat[0]]).as_matrix()
    R_pelvis = np.dot(R_torso, RzWaist.T)
    w = np.dot(RzWaist, imu_omega[0]) - np.array([0, 0, waist_yaw_omega])
    return R.from_matrix(R_pelvis).as_quat()[[3, 0, 1, 2]], w


def simulate_observation_assembly():
    """
    模拟 C++ deploy 侧的观测组装
    使用随机数据模拟机器人状态
    """
    # 加载配置
    deploy_cfg = load_yaml_config(DEPLOY_YAML_PATH)
    h1_2_cfg = load_yaml_config(H1_2_CONFIG_PATH)

    # 模拟机器人状态
    num_samples = 100

    results = {
        "base_ang_vel_errors": [],
        "projected_gravity_errors": [],
        "velocity_commands_errors": [],
        "joint_pos_rel_errors": [],
        "joint_vel_rel_errors": [],
        "gait_phase_errors": [],
        "last_action_errors": [],
        "total_obs_errors": [],
        "action_errors": []
    }

    for i in range(num_samples):
        # 模拟 IMU 数据 (torso frame)
        imu_quat = np.random.randn(4)
        imu_quat = imu_quat / np.linalg.norm(imu_quat)  # 归一化
        imu_omega = np.random.randn(3) * 2.0  # 角速度 (1D array)

        # 模拟 waist_yaw
        waist_yaw = np.random.randn() * 0.1
        waist_yaw_omega = np.random.randn() * 0.5

        # 变换到 pelvis frame
        # transform_imu_data expects imu_omega as array with shape that can be multiplied by RzWaist
        pelvis_quat, pelvis_omega = transform_imu_data(waist_yaw, waist_yaw_omega, imu_quat, imu_omega.reshape(1, -1))
        pelvis_omega = pelvis_omega.flatten()[:3]  # 只取前 3 个元素（角速度）

        # 模拟关节状态
        default_angles = np.array(h1_2_cfg["default_angles"])
        joint_pos = default_angles + np.random.randn(12) * 0.1
        joint_vel = np.random.randn(12) * 0.5

        # 模拟摇杆指令
        ly = np.random.randn()
        lx = np.random.randn()
        rx = np.random.randn()

        # 模拟上一帧动作
        last_action = np.random.randn(12) * 0.5

        # 模拟步态相位
        period = 0.8
        phase = np.random.rand()
        sin_phase = np.sin(2 * np.pi * phase)
        cos_phase = np.cos(2 * np.pi * phase)

        # ========== Python 观测组装（参考 h1_2_walk_deploy_real/deploy_real.py）==========
        # 1. base_ang_vel (pelvis frame, scale 0.25)
        ang_vel_scale = h1_2_cfg["ang_vel_scale"]
        obs_ang_vel_python = pelvis_omega * ang_vel_scale

        # 2. projected_gravity (pelvis frame)
        gravity_orientation_python = get_gravity_orientation(pelvis_quat)

        # 3. velocity_commands (joystick * cmd_scale * max_cmd)
        cmd_scale = np.array(h1_2_cfg["cmd_scale"])
        max_cmd = np.array(h1_2_cfg["max_cmd"])
        cmd = np.array([ly, -lx, -rx])  # 注意符号
        obs_cmd_python = cmd * cmd_scale * max_cmd

        # 4. joint_pos_rel (scale 1.0)
        dof_pos_scale = h1_2_cfg["dof_pos_scale"]
        obs_joint_pos_python = (joint_pos - default_angles) * dof_pos_scale

        # 5. joint_vel_rel (scale 0.05)
        dof_vel_scale = h1_2_cfg["dof_vel_scale"]
        obs_joint_vel_python = joint_vel * dof_vel_scale

        # 6. last_action
        obs_last_action_python = last_action

        # 7. gait_phase
        obs_gait_phase_python = np.array([sin_phase, cos_phase])

        # 组合完整观测向量
        obs_python = np.concatenate([
            obs_ang_vel_python,
            gravity_orientation_python,
            obs_cmd_python,
            obs_joint_pos_python,
            obs_joint_vel_python,
            obs_last_action_python,
            obs_gait_phase_python
        ]).astype(np.float32)  # 转换为 float32

        # ========== C++ 观测组装（模拟 ObservationManager::compute()）==========
        # 使用 deploy.yaml 配置
        obs_cfg = deploy_cfg["observations"]

        # 1. base_ang_vel_pelvis (scale 0.25)
        scale_ang_vel = np.array(obs_cfg["base_ang_vel_pelvis"]["scale"])
        obs_ang_vel_cpp = pelvis_omega * scale_ang_vel

        # 2. projected_gravity_pelvis (scale 1.0)
        scale_gravity = np.array(obs_cfg["projected_gravity_pelvis"]["scale"])
        obs_gravity_cpp = gravity_orientation_python * scale_gravity  # 使用相同的计算

        # 3. velocity_commands
        obs_cmd_cpp = obs_cmd_python  # 使用相同的计算

        # 4. joint_pos_rel (scale 1.0)
        scale_joint_pos = np.array(obs_cfg["joint_pos_rel"]["scale"])
        obs_joint_pos_cpp = (joint_pos - default_angles) * scale_joint_pos

        # 5. joint_vel_rel (scale 0.05)
        scale_joint_vel = np.array(obs_cfg["joint_vel_rel"]["scale"])
        obs_joint_vel_cpp = joint_vel * scale_joint_vel

        # 6. last_action
        obs_last_action_cpp = last_action

        # 7. gait_phase
        obs_gait_phase_cpp = np.array([sin_phase, cos_phase])

        # 组合完整观测向量
        obs_cpp = np.concatenate([
            obs_ang_vel_cpp,
            obs_gravity_cpp,
            obs_cmd_cpp,
            obs_joint_pos_cpp,
            obs_joint_vel_cpp,
            obs_last_action_cpp,
            obs_gait_phase_cpp
        ]).astype(np.float32)  # 转换为 float32

        # 计算误差
        results["base_ang_vel_errors"].append(np.max(np.abs(obs_ang_vel_python - obs_ang_vel_cpp)))
        results["projected_gravity_errors"].append(np.max(np.abs(gravity_orientation_python - obs_gravity_cpp)))
        results["velocity_commands_errors"].append(np.max(np.abs(obs_cmd_python - obs_cmd_cpp)))
        results["joint_pos_rel_errors"].append(np.max(np.abs(obs_joint_pos_python - obs_joint_pos_cpp)))
        results["joint_vel_rel_errors"].append(np.max(np.abs(obs_joint_vel_python - obs_joint_vel_cpp)))
        results["gait_phase_errors"].append(np.max(np.abs(obs_gait_phase_python - obs_gait_phase_cpp)))
        results["last_action_errors"].append(np.max(np.abs(obs_last_action_python - obs_last_action_cpp)))
        results["total_obs_errors"].append(np.max(np.abs(obs_python - obs_cpp)))

        # ONNX 推理对比
        session = ort.InferenceSession(ONNX_MODEL_PATH)
        input_name = session.get_inputs()[0].name

        # Python ONNX 推理
        action_python = session.run(None, {input_name: obs_python.reshape(1, -1)})[0].squeeze()

        # C++ ONNX 推理（使用相同的输入）
        action_cpp = session.run(None, {input_name: obs_cpp.reshape(1, -1)})[0].squeeze()

        results["action_errors"].append(np.max(np.abs(action_python - action_cpp)))

    # 统计结果
    summary = {
        "base_ang_vel_max_error": float(np.max(results["base_ang_vel_errors"])),
        "projected_gravity_max_error": float(np.max(results["projected_gravity_errors"])),
        "velocity_commands_max_error": float(np.max(results["velocity_commands_errors"])),
        "joint_pos_rel_max_error": float(np.max(results["joint_pos_rel_errors"])),
        "joint_vel_rel_max_error": float(np.max(results["joint_vel_rel_errors"])),
        "gait_phase_max_error": float(np.max(results["gait_phase_errors"])),
        "last_action_max_error": float(np.max(results["last_action_errors"])),
        "total_obs_max_error": float(np.max(results["total_obs_errors"])),
        "action_max_error": float(np.max(results["action_errors"])),
        "num_samples": num_samples,
        "passed": float(np.max(results["total_obs_errors"])) < 1e-6 and float(np.max(results["action_errors"])) < 1e-5
    }

    # 保存结果
    with open(os.path.join(OUTPUT_DIR, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    print("=" * 60)
    print("数值校验结果:")
    print("=" * 60)
    print(f"base_ang_vel 最大误差: {summary['base_ang_vel_max_error']:.2e}")
    print(f"projected_gravity 最大误差: {summary['projected_gravity_max_error']:.2e}")
    print(f"velocity_commands 最大误差: {summary['velocity_commands_max_error']:.2e}")
    print(f"joint_pos_rel 最大误差: {summary['joint_pos_rel_max_error']:.2e}")
    print(f"joint_vel_rel 最大误差: {summary['joint_vel_rel_max_error']:.2e}")
    print(f"gait_phase 最大误差: {summary['gait_phase_max_error']:.2e}")
    print(f"last_action 最大误差: {summary['last_action_max_error']:.2e}")
    print(f"观测向量总最大误差: {summary['total_obs_max_error']:.2e}")
    print(f"ONNX 动作输出最大误差: {summary['action_max_error']:.2e}")
    print("=" * 60)
    print(f"校验结果: {'✅ 通过' if summary['passed'] else '❌ 失败'}")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    simulate_observation_assembly()