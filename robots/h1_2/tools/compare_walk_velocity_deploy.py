#!/usr/bin/env python3
"""
H1_2 WalkVelocity 部署数值校验脚本
对标 Note 16 compare_raw_state_obs.py，为 WalkVelocity 策略设计的全面数值对比。

校验链路：
  1. 加载 rollout 捕获的"金标准"数据（qpos/qvel/obs/act）
  2. 通过 MuJoCo mj_forward 从 qpos/qvel 重建观测（模拟 C++ 部署的 torso IMU 路径）
  3. 对比重建观测 vs 保存的 rollout 观测
  4. Python ONNX 推理：在两组观测上分别推理，对比动作输出
  5. 输出逐维度、逐帧的误差指标

输出：artifacts/walk_velocity_rollout_h1_2/raw_state_obs_compare.npz + summary.json
"""

import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort


# ---------------------------------------------------------------------------
# 路径常量
# ---------------------------------------------------------------------------
ROLLOUT_NPZ = Path("/home/meme/Documents/unitree_h1/artifacts/walk_velocity_rollout_h1_2/python_rollout_full.npz")
ONNX_PATH = Path("/home/meme/Documents/unitree_h1/h1_2_mimic_deploy_real/robots/h1_2/config/policy/walk_velocity/v0/exported/policy.onnx")
MUJOCO_XML = "/home/meme/Documents/unitree_h1/unitree_mujoco/unitree_robots/h1_2/scene.xml"
OUT_DIR = Path("/home/meme/Documents/unitree_h1/artifacts/walk_velocity_rollout_h1_2")

# ---------------------------------------------------------------------------
# 策略参数（与 deploy.yaml 一致）
# ---------------------------------------------------------------------------
DEFAULT_ANGLES = np.array([0, -0.16, 0.0, 0.36, -0.2, 0.0,
                           0, -0.16, 0.0, 0.36, -0.2, 0.0], dtype=np.float32)
ACTION_SCALE = 0.25
ANG_VEL_SCALE = 0.25
DOF_POS_SCALE = 1.0
DOF_VEL_SCALE = 0.05
NUM_ACTIONS = 12
NUM_OBS = 47


# ---------------------------------------------------------------------------
# 观测重建函数
# ---------------------------------------------------------------------------

def build_mujoco_obs_sequence(data: np.lib.npyio.NpzFile) -> np.ndarray:
    """
    通过 MuJoCo mj_forward 从 qpos/qvel 重建观测。
    模拟 C++ 部署的传感器读取路径：
      1. mj_forward 更新传感器
      2. 读取 torso IMU (quat + gyro) + waist_yaw joint (pos + vel)
      3. 应用 waist_yaw 变换到 pelvis frame
      4. 组装 47 维观测

    返回值: (T, 47) float32
    """
    model = mujoco.MjModel.from_xml_path(MUJOCO_XML)
    mj_data = mujoco.MjData(model)

    # 获取传感器地址
    imu_quat_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_quat")
    imu_gyro_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
    assert imu_quat_id >= 0 and imu_gyro_id >= 0, "IMU sensor not found"
    imu_quat_adr = model.sensor_adr[imu_quat_id]
    imu_gyro_adr = model.sensor_adr[imu_gyro_id]

    qpos_rows = data["qpos"]
    qvel_rows = data["qvel"]
    cmd = data["cmd"].astype(np.float32)
    cmd_scale = data["cmd_scale"].astype(np.float32)
    meta = data["meta"]
    acts = data["act"]

    obs_seq = np.zeros((qpos_rows.shape[0], NUM_OBS), dtype=np.float32)
    prev_action = np.zeros(NUM_ACTIONS, dtype=np.float32)

    for i in range(qpos_rows.shape[0]):
        # 加载机器人状态（h1_2_handless.xml: 34 qpos, 33 qvel）
        mj_data.qpos[:] = 0
        mj_data.qvel[:] = 0
        mj_data.qpos[:] = qpos_rows[i]
        mj_data.qvel[:] = qvel_rows[i]
        mujoco.mj_forward(model, mj_data)

        # --- 读取传感器 ---
        # waist_yaw 关节传感器
        waist_yaw = float(mj_data.sensordata[12])       # jointpos index 12
        waist_yaw_omega = float(mj_data.sensordata[39])  # jointvel index 27+12

        # torso IMU
        imu_quat = np.array(mj_data.sensordata[imu_quat_adr:imu_quat_adr+4], dtype=np.float64)
        imu_omega = np.array(mj_data.sensordata[imu_gyro_adr:imu_gyro_adr+3], dtype=np.float64)

        # 腿部关节
        leg_joint_pos = np.array(mj_data.sensordata[0:12], dtype=np.float32)
        leg_joint_vel = np.array(mj_data.sensordata[27:39], dtype=np.float32)

        # --- torso → pelvis 变换 ---
        # 旋转矩阵: torso quat (w,x,y,z) → rotation matrix
        w, x, y, z = imu_quat
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

        # pelvis frame
        R_pelvis = R_torso @ RzWaist.T
        pelvis_omega = RzWaist @ imu_omega - np.array([0.0, 0.0, waist_yaw_omega], dtype=np.float64)
        pelvis_gravity = -R_pelvis[2, :]  # projected gravity

        # --- 速度指令 ---
        vel_cmd = cmd * cmd_scale

        # --- 步态相位 ---
        phase = float(meta[i, 2])
        # 对齐 C++: 当 cmd_norm < 0.1 时输出 [0, 0]
        cmd_raw = cmd  # joystick raw position
        cmd_norm = np.linalg.norm(cmd_raw)
        if cmd_norm < 0.1:
            sin_phase = 0.0
            cos_phase = 0.0
        else:
            sin_phase = np.sin(2 * np.pi * phase)
            cos_phase = np.cos(2 * np.pi * phase)

        # --- 组装 47 维观测 ---
        obs = np.zeros(NUM_OBS, dtype=np.float32)
        obs[0:3] = (pelvis_omega * ANG_VEL_SCALE).astype(np.float32)
        obs[3:6] = pelvis_gravity.astype(np.float32)
        obs[6:9] = vel_cmd
        obs[9:21] = (leg_joint_pos - DEFAULT_ANGLES) * DOF_POS_SCALE
        obs[21:33] = leg_joint_vel * DOF_VEL_SCALE
        obs[33:45] = prev_action
        obs[45] = sin_phase
        obs[46] = cos_phase
        obs_seq[i] = obs

        prev_action = acts[i].astype(np.float32)

    return obs_seq


def build_simple_obs_sequence(data: np.lib.npyio.NpzFile) -> np.ndarray:
    """
    简化的观测重建：直接使用 world-frame qpos/qvel 中的浮点基座姿态和角速度。
    ⚠️ 这种"简化"方式忽略了 torso→pelvis 变换，
       因此与 rollout 保存的 obs 应有显著差异，用于量化该变换的重要性。

    返回值: (T, 47) float32
    """
    qpos_rows = data["qpos"]
    qvel_rows = data["qvel"]
    cmd = data["cmd"].astype(np.float32)
    cmd_scale = data["cmd_scale"].astype(np.float32)
    meta = data["meta"]
    acts = data["act"]

    obs_seq = np.zeros((qpos_rows.shape[0], NUM_OBS), dtype=np.float32)
    prev_action = np.zeros(NUM_ACTIONS, dtype=np.float32)

    for i in range(qpos_rows.shape[0]):
        # World-frame 浮点基座的姿态和角速度
        quat = qpos_rows[i, 3:7]  # [w, x, y, z]
        omega = qvel_rows[i, 3:6]

        # projected gravity (world frame quat 的直接计算)
        qw, qx, qy, qz = quat.astype(np.float64)
        gravity = np.array([
            2 * (-qz * qx + qw * qy),
            -2 * (qz * qy + qw * qx),
            1 - 2 * (qw * qw + qz * qz),
        ], dtype=np.float32)

        # 腿部关节
        qj = (qpos_rows[i, 7:19] - DEFAULT_ANGLES) * DOF_POS_SCALE
        dqj = qvel_rows[i, 6:18] * DOF_VEL_SCALE

        # 指令
        vel_cmd = cmd * cmd_scale

        # 步态相位
        phase = float(meta[i, 2])
        # 对齐 C++: 当 cmd_norm < 0.1 时输出 [0, 0]
        cmd_raw = cmd
        cmd_norm = np.linalg.norm(cmd_raw)
        if cmd_norm < 0.1:
            sin_phase = 0.0
            cos_phase = 0.0
        else:
            sin_phase = np.sin(2 * np.pi * phase)
            cos_phase = np.cos(2 * np.pi * phase)

        obs = np.zeros(NUM_OBS, dtype=np.float32)
        obs[0:3] = omega * ANG_VEL_SCALE
        obs[3:6] = gravity
        obs[6:9] = vel_cmd
        obs[9:21] = qj.astype(np.float32)
        obs[21:33] = dqj.astype(np.float32)
        obs[33:45] = prev_action
        obs[45] = sin_phase
        obs[46] = cos_phase
        obs_seq[i] = obs

        prev_action = acts[i].astype(np.float32)

    return obs_seq


# ---------------------------------------------------------------------------
# ONNX 推理
# ---------------------------------------------------------------------------
def run_python_policy(obs_seq: np.ndarray, onnx_path: str) -> np.ndarray:
    """运行 Python ONNX Runtime 推理"""
    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name

    outputs = []
    for obs in obs_seq:
        act = ort_session.run(None, {input_name: obs.reshape(1, -1).astype(np.float32)})[0]
        act = act.squeeze().astype(np.float32)
        outputs.append(act)
    return np.stack(outputs, axis=0)


# ---------------------------------------------------------------------------
# 误差工具函数
# ---------------------------------------------------------------------------
def dim_summary(diff: np.ndarray, topk: int = 10) -> list[dict]:
    """输出每维最大误差排名"""
    max_per_dim = np.max(np.abs(diff), axis=0)
    idx = np.argsort(-max_per_dim)
    return [
        {"dim": int(i), "max_abs": float(max_per_dim[i])}
        for i in idx[:topk]
    ]


# ---------------------------------------------------------------------------
# 主逻辑
# ---------------------------------------------------------------------------
def main() -> None:
    data = np.load(ROLLOUT_NPZ)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    onnx_path = str(ONNX_PATH.resolve())

    # ---- 1. 获取参考观测 ----
    obs_saved = data["obs"].astype(np.float32)
    # ---- 2. 重建观测（两种路径） ----
    obs_mujoco = build_mujoco_obs_sequence(data)   # 传感器路径（对齐 C++ 部署）
    obs_simple = build_simple_obs_sequence(data)    # 简化路径（无 torso 变换）
    # ---- 3. 计算误差 ----
    mujoco_diff = obs_mujoco - obs_saved
    simple_diff = obs_simple - obs_saved

    print("=" * 70)
    print("WalkVelocity 观测重建数值校验结果")
    print("=" * 70)

    print(f"\n--- MuJoCo 传感器路径 vs 保存的 Rollout 观测 ---")
    print(f"  最大绝对误差: {np.max(np.abs(mujoco_diff)):.6e}")
    print(f"  平均绝对误差: {np.mean(np.abs(mujoco_diff)):.6e}")
    print(f"  Top-10 误差维度:")
    for d in dim_summary(mujoco_diff):
        print(f"    维度 {d['dim']:2d}: {d['max_abs']:.6e}")

    print(f"\n--- 简化路径 (world-frame, 无 torso 变换) vs 保存的 Rollout 观测 ---")
    print(f"  最大绝对误差: {np.max(np.abs(simple_diff)):.6e}")
    print(f"  平均绝对误差: {np.mean(np.abs(simple_diff)):.6e}")

    # ---- 4. ONNX 推理对比 ----
    act_saved = data["act"].astype(np.float32)

    act_from_saved = run_python_policy(obs_saved, onnx_path)
    act_from_mujoco = run_python_policy(obs_mujoco, onnx_path)
    act_from_simple = run_python_policy(obs_simple, onnx_path)

    act_mujoco_diff = act_from_mujoco - act_from_saved
    act_simple_diff = act_from_simple - act_from_saved
    act_vs_rollout_diff = act_from_saved - act_saved  # 网络输出 vs rollout 保存的 action

    print(f"\n--- 动作输出对比 ---")
    print(f"  Rollout 保存 vs MuJoCo 重建观测 → 动作差异:")
    print(f"    最大绝对误差: {np.max(np.abs(act_mujoco_diff)):.6e}")
    print(f"  Rollout 保存 vs 简化观测 → 动作差异:")
    print(f"    最大绝对误差: {np.max(np.abs(act_simple_diff)):.6e}")
    print(f"  ONNX 推理动作 vs Rollout 保存动作:")
    print(f"    最大绝对误差: {np.max(np.abs(act_vs_rollout_diff)):.6e}")

    # ---- 5. 按组件分维度误差（MuJoCo 路径） ----
    print(f"\n--- 分组件误差 (MuJoCo 路径 vs Saved Obs) ---")
    components = {
        "base_ang_vel (0-2)":     (0, 3),
        "projected_gravity (3-5)": (3, 6),
        "velocity_commands (6-8)": (6, 9),
        "joint_pos_rel (9-20)":   (9, 21),
        "joint_vel_rel (21-32)":  (21, 33),
        "last_action (33-44)":    (33, 45),
        "gait_phase (45-46)":     (45, 47),
    }
    for name, (s, e) in components.items():
        diff = mujoco_diff[:, s:e]
        print(f"  {name:30s}   max_err={np.max(np.abs(diff)):.6e}  mean_err={np.mean(np.abs(diff)):.6e}")

    # ---- 6. 目标关节角度对比 ----
    target_saved = data["target"].astype(np.float32)
    target_from_mujoco = act_from_mujoco * ACTION_SCALE + DEFAULT_ANGLES
    target_diff = target_from_mujoco - target_saved
    print(f"\n--- 目标关节角度对比 ---")
    print(f"  MuJoCo 重建路径 vs Rollout 保存:")
    print(f"    最大绝对误差: {np.max(np.abs(target_diff)):.6e}")

    # ---- 7. 保存结果 ----
    summary = {
        "steps": int(obs_saved.shape[0]),
        "policy_path": str(ONNX_PATH),
        "rollout_npz": str(ROLLOUT_NPZ),
        "mujoco_obs_vs_saved_obs_max_abs": float(np.max(np.abs(mujoco_diff))),
        "mujoco_obs_vs_saved_obs_mean_abs": float(np.mean(np.abs(mujoco_diff))),
        "mujoco_obs_vs_saved_obs_top_dims": dim_summary(mujoco_diff),
        "simple_obs_vs_saved_obs_max_abs": float(np.max(np.abs(simple_diff))),
        "simple_obs_vs_saved_obs_mean_abs": float(np.mean(np.abs(simple_diff))),
        "component_errors": {
            name: {
                "max_abs": float(np.max(np.abs(mujoco_diff[:, s:e]))),
                "mean_abs": float(np.mean(np.abs(mujoco_diff[:, s:e]))),
            }
            for name, (s, e) in components.items()
        },
        "action_diff_mujoco_obs_vs_saved_obs_max_abs": float(np.max(np.abs(act_mujoco_diff))),
        "action_diff_simple_obs_vs_saved_obs_max_abs": float(np.max(np.abs(act_simple_diff))),
        "action_diff_onnx_vs_rollout_max_abs": float(np.max(np.abs(act_vs_rollout_diff))),
        "target_diff_max_abs": float(np.max(np.abs(target_diff))),
        "first_step_saved_obs": obs_saved[0].tolist(),
        "first_step_mujoco_obs": obs_mujoco[0].tolist(),
        "first_step_saved_action": act_saved[0].tolist(),
        "first_step_onnx_action_from_saved_obs": act_from_saved[0].tolist(),
    }

    # 保存 npz
    np.savez_compressed(
        OUT_DIR / "raw_state_obs_compare.npz",
        obs_saved=obs_saved,
        obs_mujoco=obs_mujoco,
        obs_simple=obs_simple,
        act_saved=act_saved,
        act_from_saved=act_from_saved,
        act_from_mujoco=act_from_mujoco,
        act_from_simple=act_from_simple,
        target_saved=target_saved,
        target_from_mujoco=target_from_mujoco,
    )
    # 保存 summary.json
    (OUT_DIR / "summary_compare.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
