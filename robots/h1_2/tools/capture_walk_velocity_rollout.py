#!/usr/bin/env python3
"""
H1_2 WalkVelocity Python MuJoCo Rollout 捕获脚本
对标 Note 16 capture_python_rollout.py，专为 WalkVelocity 策略设计。

功能：
  - 在 MuJoCo 仿真中运行 WalkVelocity ONNX 策略
  - 通过 mj_forward 模拟 torso IMU 传感器读数
  - 应用 waist_yaw 变换到 pelvis frame（与 State_WalkVelocity.cpp 一致）
  - 保存完整的 qpos/qvel/obs/act/target 轨迹作为"金标准"参考数据

输出：artifacts/walk_velocity_rollout_h1_2/python_rollout_full.npz + summary.json
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
ONNX_PATH = Path("/home/meme/Documents/unitree_h1/h1_2_mimic_deploy_real/robots/h1_2/config/policy/walk_velocity/v0/exported/policy.onnx")
MUJOCO_XML = "/home/meme/Documents/unitree_h1/unitree_mujoco/unitree_robots/h1_2/scene.xml"
OUT_DIR_DEFAULT = "/home/meme/Documents/unitree_h1/artifacts/walk_velocity_rollout_h1_2"

# 观测/动作参数（与 deploy.yaml + h1_2.yaml 一致）
DEFAULT_ANGLES = np.array([0, -0.16, 0.0, 0.36, -0.2, 0.0,
                           0, -0.16, 0.0, 0.36, -0.2, 0.0], dtype=np.float32)
ACTION_SCALE = 0.25
ANG_VEL_SCALE = 0.25
DOF_POS_SCALE = 1.0
DOF_VEL_SCALE = 0.05
CMD_SCALE = np.array([2.0, 2.0, 0.25], dtype=np.float32)

# 默认摇杆指令 [vx, vy, omega]  — 模拟手柄前推
CMD_INIT = np.array([0.3, 0.0, 0.0], dtype=np.float32)
# 速度指令范围（用于 clamp）
MAX_CMD = np.array([0.8, 0.5, 1.57], dtype=np.float32)

NUM_ACTIONS = 12
NUM_OBS = 47
CONTROL_DECIMATION = 50  # 每 50 个子步执行一次控制（sim_dt=0.0005, control_dt=0.02）
SIMULATION_DT = 0.0005
GAIT_PERIOD = 0.8       # 步态周期（秒）
STEP_DT = 0.02          # deploy.yaml 中的控制周期（用于 gait_phase 累加）


# ---------------------------------------------------------------------------
# 观测组装（严格对齐 State_WalkVelocity.cpp）
# ---------------------------------------------------------------------------
def get_projected_gravity_pelvis(waist_yaw: float, imu_quat: np.ndarray) -> np.ndarray:
    """
    将 torso IMU 四元数变换到 pelvis frame 后计算投影重力向量。
    对齐 State_WalkVelocity.cpp get_projected_gravity_pelvis() 中的算法：
      R_pelvis = R_torso * RzWaist^T
      gravity_pelvis = R_pelvis^T * [0, 0, -1]
    """
    # imu_quat: [w, x, y, z] — MuJoCo sensor 格式
    w, x, y, z = imu_quat.astype(np.float64)

    # torso rotation matrix from quat
    R_torso = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ], dtype=np.float64)

    # RzWaist
    cz = np.cos(waist_yaw)
    sz = np.sin(waist_yaw)
    RzWaist = np.array([
        [cz, -sz, 0.0],
        [sz,  cz, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    R_pelvis = R_torso @ RzWaist.T  # 对应 State_WalkVelocity.cpp L36

    # projected gravity = -R_pelvis[2, :] (gravity direction in pelvis frame)
    gravity_pelvis = -R_pelvis[2, :]
    return gravity_pelvis.astype(np.float32)


def get_ang_vel_pelvis(waist_yaw: float, waist_yaw_omega: float,
                       imu_omega: np.ndarray) -> np.ndarray:
    """
    将 torso IMU 角速度变换到 pelvis frame。
    对齐 State_WalkVelocity.cpp get_ang_vel_pelvis() 中的算法：
      pelvis_omega = RzWaist * torso_omega
      pelvis_omega[2] -= waist_yaw_omega
    """
    imu_omega = imu_omega.astype(np.float64)

    cz = np.cos(waist_yaw)
    sz = np.sin(waist_yaw)
    RzWaist = np.array([
        [cz, -sz, 0.0],
        [sz,  cz, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    pelvis_omega = RzWaist @ imu_omega
    pelvis_omega[2] -= waist_yaw_omega  # 对应 C++ L66
    return pelvis_omega.astype(np.float32)


# ---------------------------------------------------------------------------
# 主逻辑
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="WalkVelocity Python MuJoCo Rollout 捕获")
    parser.add_argument("--steps", type=int, default=200,
                        help="控制步数（默认 200 = 4 秒 @ 50Hz）")
    parser.add_argument("--out-dir", default=OUT_DIR_DEFAULT)
    parser.add_argument("--cmd", type=float, nargs=3, default=list(CMD_INIT),
                        help="摇杆指令 vx vy omega")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = np.array(args.cmd, dtype=np.float32)

    # ---- 1. 加载 MuJoCo 模型 ----
    m = mujoco.MjModel.from_xml_path(MUJOCO_XML)
    d = mujoco.MjData(m)
    m.opt.timestep = SIMULATION_DT

    # 获取传感器地址
    imu_quat_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_quat")
    imu_gyro_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
    assert imu_quat_id >= 0 and imu_gyro_id >= 0, "IMU sensor not found"
    imu_quat_adr = m.sensor_adr[imu_quat_id]
    imu_gyro_adr = m.sensor_adr[imu_gyro_id]

    # ---- 2. 加载 ONNX 策略 ----
    onnx_path = str(ONNX_PATH.resolve())
    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name
    input_shape = ort_session.get_inputs()[0].shape
    print(f"[capture] ONNX model loaded: {ONNX_PATH.name}")
    print(f"          input: '{input_name}' shape={input_shape}")

    # ---- 3. 仿真循环 ----
    action = np.zeros(NUM_ACTIONS, dtype=np.float32)
    target_dof_pos = DEFAULT_ANGLES.copy()
    obs = np.zeros(NUM_OBS, dtype=np.float32)

    qpos_rows: list[np.ndarray] = []
    qvel_rows: list[np.ndarray] = []
    obs_rows: list[np.ndarray] = []
    act_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    tau_rows: list[np.ndarray] = []
    meta_rows: list[np.ndarray] = []

    # 获取执行器数量（h1_2_handless.xml 有 27 个执行器）
    nact = m.nu
    leg_kps = np.array([200, 200, 200, 300, 40, 40, 200, 200, 200, 300, 40, 40], dtype=np.float32)
    leg_kds = np.array([2.5, 2.5, 2.5, 4, 2, 2, 2.5, 2.5, 2.5, 4, 2, 2], dtype=np.float32)

    # 手臂/腰部 PD 参数（对齐 State_WalkVelocity.cpp 的 arm_waist_kp_ / arm_waist_kd_ / arm_waist_target_）
    # 索引顺序: [torso(1), left_arm(7), right_arm(7)] = 15 个关节
    arm_kps = np.array([300, 120, 120, 120, 80, 80, 80, 80, 120, 120, 120, 80, 80, 80, 80], dtype=np.float32)
    arm_kds = np.array([3, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1], dtype=np.float32)
    arm_target = np.zeros(15, dtype=np.float32)  # 所有手臂/腰部固定目标为 0

    counter = 0
    global_phase = 0.0  # gait_phase 累加器（对齐 C++ ObservationManager）
    while len(obs_rows) < args.steps:
        # --- PD 控制：腿部（策略输出）+ 手臂/腰部（固定目标）---
        tau_leg = leg_kps * (target_dof_pos - d.qpos[7:19]) - leg_kds * d.qvel[6:18]
        tau_arm = arm_kps * (arm_target - d.qpos[19:]) - arm_kds * d.qvel[18:]
        d.ctrl[:] = np.zeros(nact, dtype=np.float64)
        d.ctrl[:12] = tau_leg
        d.ctrl[12:] = tau_arm
        mujoco.mj_step(m, d)
        counter += 1

        if counter % CONTROL_DECIMATION != 0:
            continue

        # --- mj_forward 更新传感器 ---
        mujoco.mj_forward(m, d)

        # --- 读取 IMU 传感器 (torso frame) ---
        imu_quat = np.array(d.sensordata[imu_quat_adr:imu_quat_adr+4], dtype=np.float64)
        imu_omega = np.array(d.sensordata[imu_gyro_adr:imu_gyro_adr+3], dtype=np.float64)

        # --- 读取 waist_yaw 关节 ---
        waist_yaw = float(d.sensordata[12])       # jointpos sensor at index 12
        waist_yaw_omega = float(d.sensordata[39])  # jointvel sensor at index 27+12

        # --- Transform torso → pelvis ---
        pelvis_gravity = get_projected_gravity_pelvis(waist_yaw, imu_quat)
        pelvis_omega = get_ang_vel_pelvis(waist_yaw, waist_yaw_omega, imu_omega)

        # --- 读取腿部关节传感器 ---
        leg_joint_pos = np.array(d.sensordata[0:12], dtype=np.float32)     # jointpos 0-11
        leg_joint_vel = np.array(d.sensordata[27:39], dtype=np.float32)    # jointvel 27-38

        # --- 速度指令（joystick position × CMD_SCALE） ---
        # 对齐 C++ velocity_commands(): clamp(ly, -range, range) × scale
        # 此处 cmd 为 joystick 原始值（[-1, 1] 区间），直接乘以 scale
        vel_cmd = cmd * CMD_SCALE

        # --- 步态相位（对齐 C++ ObservationManager gait_phase） ---
        # C++ 代码: env->global_phase += env->step_dt * (1.0 / period)
        #            cmd = velocity_commands(env)  # 返回 raw clamping 值（before scale）
        #            if norm(cmd) < 0.1: output [0, 0]
        cmd_norm = np.linalg.norm(cmd)  # raw joystick 值，对齐 C++
        delta_phase = STEP_DT * (1.0 / GAIT_PERIOD)
        global_phase = (global_phase + delta_phase) % 1.0
        sin_phase = np.sin(2 * np.pi * global_phase)
        cos_phase = np.cos(2 * np.pi * global_phase)
        if cmd_norm < 0.1:
            sin_phase = 0.0
            cos_phase = 0.0

        # --- 组装 47 维观测 ---
        obs[0:3] = pelvis_omega * ANG_VEL_SCALE          # base_ang_vel_pelvis * scale 0.25
        obs[3:6] = pelvis_gravity                         # projected_gravity_pelvis
        obs[6:9] = vel_cmd                                # velocity_commands
        obs[9:21] = (leg_joint_pos - DEFAULT_ANGLES) * DOF_POS_SCALE   # joint_pos_rel
        obs[21:33] = leg_joint_vel * DOF_VEL_SCALE        # joint_vel_rel
        obs[33:45] = action                               # last_action
        obs[45] = sin_phase                               # gait_phase sin
        obs[46] = cos_phase                               # gait_phase cos

        # --- ONNX 推理 ---
        obs_input = obs.reshape(1, -1).astype(np.float32)
        act_raw = ort_session.run(None, {input_name: obs_input})[0].squeeze().astype(np.float32)
        action = act_raw.copy()
        target_dof_pos = act_raw * ACTION_SCALE + DEFAULT_ANGLES

        # --- 保存数据 ---
        qpos_rows.append(d.qpos.copy().astype(np.float32))
        qvel_rows.append(d.qvel.copy().astype(np.float32))
        obs_rows.append(obs.copy())
        act_rows.append(action.copy())
        target_rows.append(target_dof_pos.copy())
        tau_rows.append(np.concatenate([tau_leg, tau_arm]).astype(np.float32))
        meta_rows.append(np.array([counter, d.time, global_phase], dtype=np.float64))

    # ---- 4. 保存结果 ----
    data_dict = dict(
        qpos=np.stack(qpos_rows, axis=0),
        qvel=np.stack(qvel_rows, axis=0),
        obs=np.stack(obs_rows, axis=0),
        act=np.stack(act_rows, axis=0),
        target=np.stack(target_rows, axis=0),
        tau=np.stack(tau_rows, axis=0),
        meta=np.stack(meta_rows, axis=0),
        default_angles=DEFAULT_ANGLES,
        cmd=cmd,
        cmd_scale=CMD_SCALE,
    )
    np.savez_compressed(out_dir / "python_rollout_full.npz", **data_dict)

    summary = {
        "steps": len(obs_rows),
        "policy": str(ONNX_PATH),
        "xml": MUJOCO_XML,
        "npz": str(out_dir / "python_rollout_full.npz"),
        "num_obs": NUM_OBS,
        "num_actions": NUM_ACTIONS,
        "action_scale": ACTION_SCALE,
        "cmd": cmd.tolist(),
        "cmd_scale": CMD_SCALE.tolist(),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[capture] Rollout 完成，共 {len(obs_rows)} 步，保存至 {out_dir}")


if __name__ == "__main__":
    main()
