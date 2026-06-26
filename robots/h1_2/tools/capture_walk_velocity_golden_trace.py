#!/usr/bin/env python3
"""
SKILL Step 2: Python 侧采集"黄金基准" (Golden Trace)

在 MuJoCo 中运行 WalkVelocity ONNX 策略，逐帧记录 A-F 全链路中间变量。
按照 deploy-numerical-validation SKILL 的断点定义：

  Input: raw_imu_quat, raw_imu_gyro, raw_motor_q, raw_motor_dq, raw_joystick
  A:     base_ang_vel_pelvis_raw, projected_gravity_pelvis_raw
  B:     obs_raw (47-dim, 未乘 scale/clip)
  C:     obs_scaled (47-dim, 乘 scale 后 → 送入 ONNX)
  D:     nn_output (12-dim, ONNX 原始输出)
  E:     (无额外后处理, E == D)
  F:     processed_action (12-dim, raw * scale + offset)

输出: golden_trace.npz
"""

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
ONNX_PATH = Path("/home/meme/Documents/unitree_h1/h1_2_mimic_deploy_real/robots/h1_2/"
                 "config/policy/walk_velocity/v0/exported/policy.onnx")
MUJOCO_XML = "/home/meme/Documents/unitree_h1/unitree_mujoco/unitree_robots/h1_2/scene.xml"
OUT_DIR = "/home/meme/Documents/unitree_h1/artifacts/walk_velocity_openloop"

DEFAULT_ANGLES = np.array([0, -0.16, 0.0, 0.36, -0.2, 0.0,
                           0, -0.16, 0.0, 0.36, -0.2, 0.0], dtype=np.float32)
ACTION_SCALE = np.array([0.25] * 12, dtype=np.float32)
ACTION_OFFSET = np.array([0, -0.16, 0.0, 0.36, -0.2, 0.0,
                          0, -0.16, 0.0, 0.36, -0.2, 0.0], dtype=np.float32)
ANG_VEL_SCALE = np.array([0.25, 0.25, 0.25], dtype=np.float32)
DOF_VEL_SCALE = np.array([0.05] * 12, dtype=np.float32)
CMD_SCALE = np.array([2.0, 2.0, 0.25], dtype=np.float32)
MAX_CMD = np.array([0.8, 0.5, 1.57], dtype=np.float32)

NUM_ACTIONS = 12
NUM_OBS = 47
CONTROL_DECIMATION = 50
SIMULATION_DT = 0.0005
GAIT_PERIOD = 0.8
STEP_DT = 0.02


# ---------------------------------------------------------------------------
# torso IMU → pelvis 变换（对齐 State_WalkVelocity.cpp / observations.h）
# ---------------------------------------------------------------------------
def get_ang_vel_pelvis(waist_yaw, waist_yaw_omega, imu_omega):
    """Rz(waist_yaw) @ torso_omega, [2] -= waist_yaw_omega"""
    cz, sz = np.cos(waist_yaw), np.sin(waist_yaw)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    pelvis_omega = Rz @ imu_omega.astype(np.float64)
    pelvis_omega[2] -= waist_yaw_omega
    return pelvis_omega.astype(np.float32)


def get_projected_gravity_pelvis(waist_yaw, imu_quat):
    """R_pelvis = R_torso * RzWaist^T, gravity = R_pelvis^T * [0,0,-1]"""
    w, x, y, z = imu_quat.astype(np.float64)
    R_torso = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ], dtype=np.float64)
    cz, sz = np.cos(waist_yaw), np.sin(waist_yaw)
    RzWaist = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    R_pelvis = R_torso @ RzWaist.T
    gravity_pelvis = -R_pelvis[2, :]
    return gravity_pelvis.astype(np.float32)


# ---------------------------------------------------------------------------
# 主逻辑
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="WalkVelocity Golden Trace 采集")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--cmd", type=float, nargs=3, default=[0.3, 0.0, 0.0])
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = np.array(args.cmd, dtype=np.float32)

    # ---- 1. 加载 MuJoCo 模型 ----
    m = mujoco.MjModel.from_xml_path(MUJOCO_XML)
    d = mujoco.MjData(m)
    m.opt.timestep = SIMULATION_DT

    imu_quat_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_quat")
    imu_gyro_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
    imu_quat_adr = m.sensor_adr[imu_quat_id]
    imu_gyro_adr = m.sensor_adr[imu_gyro_id]

    # ---- 2. 加载 ONNX 策略 ----
    ort_session = ort.InferenceSession(str(ONNX_PATH.resolve()))
    input_name = ort_session.get_inputs()[0].name

    # ---- 3. 仿真循环 ----
    action = np.zeros(NUM_ACTIONS, dtype=np.float32)
    target_dof_pos = DEFAULT_ANGLES.copy()
    global_phase = 0.0

    leg_kps = np.array([200, 200, 200, 300, 40, 40, 200, 200, 200, 300, 40, 40], dtype=np.float64)
    leg_kds = np.array([2.5, 2.5, 2.5, 4, 2, 2, 2.5, 2.5, 2.5, 4, 2, 2], dtype=np.float64)
    arm_kps = np.array([300, 120, 120, 120, 80, 80, 80, 80, 120, 120, 120, 80, 80, 80, 80], dtype=np.float64)
    arm_kds = np.array([3, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1], dtype=np.float64)
    arm_target = np.zeros(15, dtype=np.float64)

    # 存储 A-F 全链路中间变量
    lists = {k: [] for k in [
        'raw_imu_quat', 'raw_imu_gyro', 'raw_motor_q', 'raw_motor_dq',
        'raw_waist_yaw', 'raw_waist_yaw_omega',
        'raw_joystick',
        'ang_vel_pelvis_raw', 'gravity_pelvis_raw',
        'obs_raw', 'obs_scaled',
        'nn_output',
        'processed_action',
    ]}

    counter = 0
    while len(lists['obs_raw']) < args.steps:
        # PD 控制
        tau_leg = leg_kps * (target_dof_pos - d.qpos[7:19]) - leg_kds * d.qvel[6:18]
        tau_arm = arm_kps * (arm_target - d.qpos[19:]) - arm_kds * d.qvel[18:]
        d.ctrl[:12] = tau_leg
        d.ctrl[12:] = tau_arm
        mujoco.mj_step(m, d)
        counter += 1

        if counter % CONTROL_DECIMATION != 0:
            continue

        mujoco.mj_forward(m, d)

        # ===== Input: 读取原始传感器数据 =====
        imu_quat = np.array(d.sensordata[imu_quat_adr:imu_quat_adr+4], dtype=np.float32)
        imu_omega = np.array(d.sensordata[imu_gyro_adr:imu_gyro_adr+3], dtype=np.float32)
        leg_q = np.array(d.sensordata[0:12], dtype=np.float32)
        leg_dq = np.array(d.sensordata[27:39], dtype=np.float32)
        waist_yaw = float(d.sensordata[12])
        waist_yaw_omega = float(d.sensordata[39])

        # ===== A: 特殊状态计算 (torso → pelvis) =====
        ang_vel_pelvis = get_ang_vel_pelvis(waist_yaw, waist_yaw_omega, imu_omega)
        gravity_pelvis = get_projected_gravity_pelvis(waist_yaw, imu_quat)

        # ===== B: 观测向量构造 (未乘 scale) =====
        vel_cmd_raw = cmd * MAX_CMD  # velocity_commands 观测函数的原始输出
        delta_phase = STEP_DT * (1.0 / GAIT_PERIOD)
        global_phase = (global_phase + delta_phase) % 1.0

        obs_raw = np.zeros(NUM_OBS, dtype=np.float32)
        obs_raw[0:3]   = ang_vel_pelvis           # base_ang_vel_pelvis (raw)
        obs_raw[3:6]   = gravity_pelvis            # projected_gravity_pelvis (raw)
        obs_raw[6:9]   = vel_cmd_raw               # velocity_commands (raw)
        obs_raw[9:21]  = (leg_q - DEFAULT_ANGLES)  # joint_pos_rel (raw, scale=1)
        obs_raw[21:33] = leg_dq                     # joint_vel_rel (raw, before *0.05)
        obs_raw[33:45] = action                     # last_action (raw, scale=1)
        obs_raw[45]    = np.sin(2 * np.pi * global_phase)  # gait_phase (raw, scale=1)
        obs_raw[46]    = np.cos(2 * np.pi * global_phase)

        # ===== C: 预处理 (乘 scale) =====
        obs_scaled = obs_raw.copy()
        obs_scaled[0:3]   *= ANG_VEL_SCALE      # * 0.25
        obs_scaled[21:33] *= DOF_VEL_SCALE       # * 0.05
        obs_scaled[6:9]   *= CMD_SCALE           # * [2, 2, 0.25]
        # 其余分量 scale=1，不变

        # ===== D: 模型推理 (ONNX) =====
        obs_input = obs_scaled.reshape(1, -1).astype(np.float32)
        nn_output = ort_session.run(None, {input_name: obs_input})[0].squeeze().astype(np.float32)

        # ===== E: 后处理 (无，E == D) =====

        # ===== F: 动作计算/控制律 =====
        processed = nn_output * ACTION_SCALE + ACTION_OFFSET

        # ===== 保存全量中间变量 =====
        lists['raw_imu_quat'].append(imu_quat)
        lists['raw_imu_gyro'].append(imu_omega)
        lists['raw_motor_q'].append(leg_q)
        lists['raw_motor_dq'].append(leg_dq)
        lists['raw_waist_yaw'].append(waist_yaw)
        lists['raw_waist_yaw_omega'].append(waist_yaw_omega)
        lists['raw_joystick'].append(cmd.copy())
        lists['ang_vel_pelvis_raw'].append(ang_vel_pelvis)
        lists['gravity_pelvis_raw'].append(gravity_pelvis)
        lists['obs_raw'].append(obs_raw)
        lists['obs_scaled'].append(obs_scaled)
        lists['nn_output'].append(nn_output)
        lists['processed_action'].append(processed)

        # 更新 action 和 target（闭环）
        action = nn_output.copy()
        target_dof_pos = processed.copy()

    # ---- 4. 保存 ----
    data_dict = {k: np.array(v, dtype=np.float32) for k, v in lists.items()}
    data_dict['default_joint_pos'] = DEFAULT_ANGLES
    data_dict['action_scale'] = ACTION_SCALE
    data_dict['action_offset'] = ACTION_OFFSET
    data_dict['cmd'] = cmd

    np.savez_compressed(out_dir / "golden_trace.npz", **data_dict)

    summary = {
        "steps": len(lists['obs_raw']),
        "policy": str(ONNX_PATH),
        "xml": MUJOCO_XML,
        "cmd": cmd.tolist(),
        "output": str(out_dir / "golden_trace.npz"),
    }
    (out_dir / "golden_trace_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"[OK] Golden trace 保存至 {out_dir / 'golden_trace.npz'}")


if __name__ == "__main__":
    main()
