#!/usr/bin/env python3
"""
SKILL Step 4: 自动化比对脚本

对比 golden_trace.npz (Python 基准) 和 cpp_trace.npz (C++ 回放结果)。

比对层级:
  C: obs_scaled   — 预处理后的观测向量 (47-dim)
  D: nn_output    — ONNX 推理输出 (12-dim)
  F: processed_action — 处理后的动作 (12-dim)

对于 C 层，额外反推 raw obs (B 层) 进行逐分量对比。

阈值 (SKILL 默认):
  obs_raw 反推: 1e-5 (float32 精度)
  obs_scaled:   1e-5
  nn_output:    1e-4 (ONNX Runtime 跨平台精度)
  processed_action: 1e-5
"""

import json
import sys
from pathlib import Path

import numpy as np

# 观测 scale 配置 (从 deploy.yaml 提取)
OBS_SCALES = np.array(
    [0.25, 0.25, 0.25,   # base_ang_vel_pelvis
     1.0, 1.0, 1.0,       # projected_gravity_pelvis
     2.0, 2.0, 0.25,      # velocity_commands
     *([1.0]*12),          # joint_pos_rel
     *([0.05]*12),         # joint_vel_rel
     *([1.0]*12),          # last_action
     1.0, 1.0],            # gait_phase
    dtype=np.float32
)

OBS_NAMES = (
    ["base_ang_vel_pelvis"] * 3 +
    ["projected_gravity_pelvis"] * 3 +
    ["velocity_commands"] * 3 +
    ["joint_pos_rel"] * 12 +
    ["joint_vel_rel"] * 12 +
    ["last_action"] * 12 +
    ["gait_phase"] * 2
)

THRESHOLDS = {
    "obs_raw_reverse": 1e-5,
    "obs_scaled": 1e-5,
    "nn_output": 1e-4,
    "processed_action": 1e-5,
    "gait_phase": 1e-4,  # float32 累加精度 (Python float64 vs C++ float32)
}


def compare_layer(name, expected, actual, threshold):
    """对比单层，返回 (pass, max_err, mean_err)"""
    diff = np.abs(expected - actual)
    max_err = float(np.max(diff))
    mean_err = float(np.mean(diff))
    passed = max_err < threshold
    return passed, max_err, mean_err


def main():
    if len(sys.argv) < 2:
        print("用法: python compare_traces.py <artifacts_dir>")
        sys.exit(1)

    art_dir = Path(sys.argv[1])
    golden_path = art_dir / "golden_trace.npz"
    cpp_path = art_dir / "cpp_trace.npz"

    golden = dict(np.load(golden_path))
    cpp = dict(np.load(cpp_path))

    n_steps = golden["obs_scaled"].shape[0]
    print(f"{'='*70}")
    print(f"  SKILL 开环注入校验 — WalkVelocity")
    print(f"  Golden: {golden_path}  ({n_steps} steps)")
    print(f"  C++:    {cpp_path}")
    print(f"{'='*70}")

    all_pass = True
    results = {}

    # ---- C: obs_scaled 对比 ----
    py_obs = golden["obs_scaled"].reshape(n_steps, -1)
    cpp_obs = cpp["obs_scaled"].reshape(n_steps, -1)

    gait_phase_mask = np.array([n == "gait_phase" for n in OBS_NAMES])
    non_gp_mask = ~gait_phase_mask

    c_gp_max = float(np.max(np.abs(py_obs[:, gait_phase_mask] - cpp_obs[:, gait_phase_mask])))
    c_ngp_diff = np.abs(py_obs[:, non_gp_mask] - cpp_obs[:, non_gp_mask])
    c_ngp_max = float(np.max(c_ngp_diff))
    c_ngp_mean = float(np.mean(c_ngp_diff))

    passed_c_gp = c_gp_max < THRESHOLDS["gait_phase"]
    passed_c_ngp = c_ngp_max < THRESHOLDS["obs_scaled"]
    passed_c = passed_c_gp and passed_c_ngp
    all_pass &= passed_c

    results["C:obs_scaled(non_gait_phase)"] = {"pass": passed_c_ngp, "max_err": c_ngp_max,
                                                 "mean_err": c_ngp_mean, "threshold": THRESHOLDS["obs_scaled"]}
    results["C:obs_scaled(gait_phase)"] = {"pass": passed_c_gp, "max_err": c_gp_max,
                                             "mean_err": float(np.mean(np.abs(py_obs[:, gait_phase_mask] - cpp_obs[:, gait_phase_mask]))),
                                             "threshold": THRESHOLDS["gait_phase"]}

    status_c_ngp = "PASS" if passed_c_ngp else "FAIL"
    status_c_gp = "PASS" if passed_c_gp else "FAIL"
    print(f"\n  [C] obs_scaled (non-gait): {status_c_ngp}  max={c_ngp_max:.2e}  mean={c_ngp_mean:.2e}  (阈值 {THRESHOLDS['obs_scaled']:.0e})")
    print(f"  [C] obs_scaled (gait_phase): {status_c_gp}  max={c_gp_max:.2e}  (阈值 {THRESHOLDS['gait_phase']:.0e})")

    # ---- B: obs_raw 反推对比 ----
    py_obs_raw = py_obs / OBS_SCALES
    cpp_obs_raw = cpp_obs / OBS_SCALES

    # 逐分量检查，gait_phase 使用独立阈值 (float32 累加漂移)
    gait_phase_mask = np.array([n == "gait_phase" for n in OBS_NAMES])
    non_gp_mask = ~gait_phase_mask

    gp_max = float(np.max(np.abs(py_obs_raw[:, gait_phase_mask] - cpp_obs_raw[:, gait_phase_mask])))
    non_gp_diff = np.abs(py_obs_raw[:, non_gp_mask] - cpp_obs_raw[:, non_gp_mask])
    non_gp_max = float(np.max(non_gp_diff)) if non_gp_diff.size > 0 else 0.0
    non_gp_mean = float(np.mean(non_gp_diff)) if non_gp_diff.size > 0 else 0.0

    passed_gp = gp_max < THRESHOLDS["gait_phase"]
    passed_non_gp = non_gp_max < THRESHOLDS["obs_raw_reverse"]
    passed_raw = passed_gp and passed_non_gp
    all_pass &= passed_raw

    results["B:obs_raw(non_gait_phase)"] = {"pass": passed_non_gp, "max_err": non_gp_max,
                                              "mean_err": non_gp_mean, "threshold": THRESHOLDS["obs_raw_reverse"]}
    results["B:obs_raw(gait_phase)"] = {"pass": passed_gp, "max_err": gp_max,
                                         "mean_err": float(np.mean(np.abs(py_obs_raw[:, gait_phase_mask] - cpp_obs_raw[:, gait_phase_mask]))),
                                         "threshold": THRESHOLDS["gait_phase"]}

    status_gp = "PASS" if passed_gp else "FAIL"
    status_ngp = "PASS" if passed_non_gp else "FAIL"
    print(f"  [B] obs_raw  (non-gait):  {status_ngp}  max={non_gp_max:.2e}  mean={non_gp_mean:.2e}  (阈值 {THRESHOLDS['obs_raw_reverse']:.0e})")
    print(f"  [B] obs_raw  (gait_phase): {status_gp}  max={gp_max:.2e}  (阈值 {THRESHOLDS['gait_phase']:.0e}, float32累加漂移)")

    # 逐分量报告 obs_raw 差异
    if not passed_non_gp:
        print(f"\n  {'分量':<28s}  {'max_err':>10s}  {'mean_err':>10s}  {'step':>6s}")
        print(f"  {'-'*28}  {'-'*10}  {'-'*10}  {'-'*6}")
        for i, comp_name in enumerate(OBS_NAMES):
            if comp_name == "gait_phase":
                continue
            comp_diff = np.abs(py_obs_raw[:, i] - cpp_obs_raw[:, i])
            c_max = float(np.max(comp_diff))
            if c_max > 1e-7:
                c_step = int(np.argmax(comp_diff))
                print(f"  {comp_name:<28s}  {c_max:10.2e}  {float(np.mean(comp_diff)):10.2e}  {c_step:6d}")

    # ---- D: nn_output 对比 ----
    py_nn = golden["nn_output"].reshape(n_steps, -1)
    cpp_nn = cpp["nn_output"].reshape(n_steps, -1)
    passed_nn, max_e_nn, mean_e_nn = compare_layer("nn_output", py_nn, cpp_nn, THRESHOLDS["nn_output"])
    all_pass &= passed_nn
    results["D:nn_output"] = {"pass": passed_nn, "max_err": max_e_nn, "mean_err": mean_e_nn,
                               "threshold": THRESHOLDS["nn_output"]}
    status = "PASS" if passed_nn else "FAIL"
    print(f"\n  [D] nn_output (12-dim): {status}  max={max_e_nn:.2e}  mean={mean_e_nn:.2e}  (阈值 {THRESHOLDS['nn_output']:.0e})")

    # ---- F: processed_action 对比 ----
    py_act = golden["processed_action"].reshape(n_steps, -1)
    cpp_act = cpp["processed_action"].reshape(n_steps, -1)
    passed_act, max_e_act, mean_e_act = compare_layer("processed_action", py_act, cpp_act, THRESHOLDS["processed_action"])
    all_pass &= passed_act
    results["F:processed_action"] = {"pass": passed_act, "max_err": max_e_act, "mean_err": mean_e_act,
                                      "threshold": THRESHOLDS["processed_action"]}
    status = "PASS" if passed_act else "FAIL"
    print(f"  [F] processed_action:   {status}  max={max_e_act:.2e}  mean={mean_e_act:.2e}  (阈值 {THRESHOLDS['processed_action']:.0e})")

    # ---- 总结 ----
    print(f"\n{'='*70}")
    if all_pass:
        print("  ✓ 全链路 PASS — C++ 部署管线与 Python 基准数值对齐")
    else:
        print("  ✗ 存在 FAIL 项 — 需要排查修复")
    print(f"{'='*70}")

    # 保存结果
    report = {
        "steps": n_steps,
        "all_pass": all_pass,
        "layers": results,
    }
    (art_dir / "comparison_report.json").write_text(
        json.dumps(report, indent=2) + "\n")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
