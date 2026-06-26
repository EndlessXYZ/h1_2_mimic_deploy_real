#!/usr/bin/env python3
"""
Analyze C++ Mimic dump CSV (State_Mimic output) and compare against
Python training-side observation assembly formulas.

CSV columns:
  step, episode_length,
  obs_0 .. obs_143  (144-dim observation),
  raw_action_0 .. raw_action_26  (27 ONNX outputs),
  processed_action_0 .. processed_action_26  (27 scaled actions),
  motor_q_0 .. motor_q_26  (27 motor positions),
  motor_dq_0 .. motor_dq_26  (27 motor velocities)

Usage:
  python analyze_mimic_cpp_dump.py \
      --csv /tmp/mimic_dump.csv \
      --motion path/to/motion.npz \
      --deploy-yaml path/to/deploy.yaml

Output:
  - Summary printed to stdout
  - JSON saved to --output-dir
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_CSV = "/tmp/mimic_dump.csv"
OUTPUT_DIR = Path("/home/meme/Documents/unitree_h1/artifacts/mimic_cpp_dump_analysis")

NUM_OBS = 144
NUM_ACTIONS = 27
NUM_MOTORS = 27

# Observation layout (half-open ranges) — No-State-Estimation 144-dim
OBS_SEGMENTS = {
    "motion_ref_pos":       (0, 27),
    "motion_ref_vel":       (27, 54),
    "motion_anchor_ori_b":  (54, 60),
    "base_ang_vel":         (60, 63),
    "joint_pos_rel":        (63, 90),
    "joint_vel_rel":        (90, 117),
    "last_action":          (117, 144),
}

# No-State-Estimation: no known zero-placeholder segments

# ---------------------------------------------------------------------------
# Quaternion / rotation helpers (w, x, y, z convention)
# ---------------------------------------------------------------------------

def quat_multiply(q1, q2):
    """Quaternion multiplication (w, x, y, z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64)


def quat_conjugate(q):
    """Quaternion conjugate (w, x, y, z)."""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_to_rotmat(q):
    """Quaternion -> 3x3 rotation matrix (w, x, y, z)."""
    w, x, y, z = q.astype(np.float64)
    return np.array([
        [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)],
    ], dtype=np.float64)


def rotmat_to_6d(R):
    """Rotation matrix -> 6D representation (first two columns)."""
    return np.array([
        R[0, 0], R[0, 1], R[1, 0], R[1, 1], R[2, 0], R[2, 1],
    ], dtype=np.float32)


def yaw_quaternion(q):
    """Extract yaw component from quaternion as (w, 0, 0, z)."""
    R = quat_to_rotmat(q)
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)], dtype=np.float64)


def compute_torso_quat(pelvis_quat, waist_yaw_angle):
    """Compute torso quaternion: pelvis_quat * Rz(waist_yaw)."""
    half = waist_yaw_angle / 2
    torso_yaw_q = np.array(
        [np.cos(half), 0, 0, np.sin(half)], dtype=np.float64
    )
    return quat_multiply(pelvis_quat.astype(np.float64), torso_yaw_q)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_dump(path: str) -> pd.DataFrame:
    """Load C++ dump CSV."""
    p = Path(path)
    if not p.exists():
        print(f"[ERROR] CSV file not found: {path}")
        sys.exit(1)
    df = pd.read_csv(p)
    print(f"Loaded {len(df)} rows from {path}")
    return df


def load_motion(path: str) -> dict:
    """Load motion NPZ file."""
    p = Path(path)
    if not p.exists():
        print(f"[ERROR] Motion file not found: {path}")
        sys.exit(1)
    data = np.load(p)
    result = {
        "joint_pos": data["joint_pos"],      # (N, 27)
        "joint_vel": data["joint_vel"],      # (N, 27)
        "body_quat_w": data["body_quat_w"],  # (N, B, 4)
    }
    if "body_pos_w" in data:
        result["body_pos_w"] = data["body_pos_w"]  # (N, B, 3)
    if "fps" in data:
        result["fps"] = float(data["fps"])
    print(f"Loaded motion: {result['joint_pos'].shape[0]} frames, "
          f"{result['joint_pos'].shape[1]} joints")
    return result


def load_deploy_yaml(path: str) -> dict:
    """Load deploy.yaml and extract scale/offset/default_joint_pos."""
    p = Path(path)
    if not p.exists():
        print(f"[ERROR] deploy.yaml not found: {path}")
        sys.exit(1)
    with open(p, "r") as f:
        cfg = yaml.safe_load(f)

    default_joint_pos = np.array(cfg["default_joint_pos"], dtype=np.float32)

    action_cfg = cfg.get("actions", {}).get("JointPositionAction", {})
    scale = np.array(action_cfg.get("scale", [1.0] * NUM_ACTIONS), dtype=np.float32)
    offset = np.array(action_cfg.get("offset", [0.0] * NUM_ACTIONS), dtype=np.float32)

    print(f"Loaded deploy.yaml: {len(default_joint_pos)} default joints, "
          f"scale shape={scale.shape}, offset shape={offset.shape}")
    return {
        "default_joint_pos": default_joint_pos,
        "action_scale": scale,
        "action_offset": offset,
    }


# ---------------------------------------------------------------------------
# Array extraction from DataFrame
# ---------------------------------------------------------------------------

def extract_obs(df: pd.DataFrame) -> np.ndarray:
    cols = [f"obs_{i}" for i in range(NUM_OBS)]
    return df[cols].values.astype(np.float32)


def extract_raw_actions(df: pd.DataFrame) -> np.ndarray:
    cols = [f"raw_action_{i}" for i in range(NUM_ACTIONS)]
    return df[cols].values.astype(np.float32)


def extract_processed_actions(df: pd.DataFrame) -> np.ndarray:
    cols = [f"processed_action_{i}" for i in range(NUM_ACTIONS)]
    return df[cols].values.astype(np.float32)


def extract_motor_q(df: pd.DataFrame) -> np.ndarray:
    cols = [f"motor_q_{i}" for i in range(NUM_MOTORS)]
    return df[cols].values.astype(np.float32)


def extract_motor_dq(df: pd.DataFrame) -> np.ndarray:
    cols = [f"motor_dq_{i}" for i in range(NUM_MOTORS)]
    return df[cols].values.astype(np.float32)


# ---------------------------------------------------------------------------
# Reconstruct Python training-side observations from CSV data
# ---------------------------------------------------------------------------

def reconstruct_training_obs(
    motor_q: np.ndarray,          # (T, 27)
    motor_dq: np.ndarray,         # (T, 27)
    raw_actions: np.ndarray,      # (T, 27)
    motion: dict,
    default_joint_pos: np.ndarray,
) -> np.ndarray:
    """
    Reconstruct the 144-dim observation (No-State-Estimation) as the Python training side would
    assemble it, using data from the C++ dump.

    For each step t:
      [0:27]   motion_ref_pos       = motion.joint_pos[frame]
      [27:54]  motion_ref_vel       = motion.joint_vel[frame]
      [54:60]  motion_anchor_ori_b  = relative orientation 6D
      [60:63]  base_ang_vel         = IMU gyro in pelvis frame
      [63:90]  joint_pos_rel        = motor_q[t] - default_joint_pos
      [90:117] joint_vel_rel        = motor_dq[t]
      [117:144] last_action         = raw_actions[t-1]  (zero for step 0)

    For motion_anchor_ori_b, we reconstruct using:
      - motor_q[t, 12] as waist_yaw to compute torso quat
      - motion body_quat_w for reference torso quat
      - init_quat from step 0 yaw alignment
      - formula: rot = ((init_quat * ref_torso_quat).conjugate() * robot_torso_quat).to_rotmat().T
        then take first 2 columns as 6D

    Note: Since we don't have the robot pelvis quaternion from the CSV,
    we use identity as the pelvis quat (the C++ side also uses IMU quat
    which approximates the torso orientation). For the purpose of comparing
    against the C++ obs, this reconstruction uses the same torso quat
    approximation that C++ uses (waist_yaw only, no pelvis quat from IMU).
    """
    T = motor_q.shape[0]
    motion_joint_pos = motion["joint_pos"]     # (N, 27)
    motion_joint_vel = motion["joint_vel"]     # (N, 27)
    motion_body_quat_w = motion["body_quat_w"] # (N, B, 4)
    num_motion_frames = motion_joint_pos.shape[0]

    obs_recon = np.zeros((T, NUM_OBS), dtype=np.float32)

    # Compute init_quat from step 0
    # Use pelvis quat (IMU quat) to match C++ fix
    # Robot pelvis quat at step 0 is identity (from IMU)
    # Motion pelvis quat at frame 0 from motion file
    robot_pelvis_quat_0 = np.array([1, 0, 0, 0], dtype=np.float64)
    motion_pelvis_quat_0 = motion_body_quat_w[0, 0].astype(np.float64)  # pelvis = body 0

    robot_yaw_0 = yaw_quaternion(robot_pelvis_quat_0)
    ref_yaw_0 = yaw_quaternion(motion_pelvis_quat_0)
    init_quat = quat_multiply(robot_yaw_0, quat_conjugate(ref_yaw_0))

    for t in range(T):
        frame = min(t, num_motion_frames - 1)  # clamp frame index

        # [0:27] motion_ref_pos
        obs_recon[t, 0:27] = motion_joint_pos[frame].astype(np.float32)

        # [27:54] motion_ref_vel
        obs_recon[t, 27:54] = motion_joint_vel[frame].astype(np.float32)

        # [54:60] motion_anchor_ori_b
        # Use pelvis quat (IMU quat) to match C++ fix
        # Reference pelvis quat from motion
        m_pelvis_q = motion_body_quat_w[frame, 0].astype(np.float64)

        # Robot pelvis quat: identity (from IMU, we don't have real IMU in CSV)
        r_pelvis_q = np.array([1, 0, 0, 0], dtype=np.float64)

        # C++ formula: (init_quat * ref)^{-1} * real, then rot.T, then 6D
        aligned_ref = quat_multiply(init_quat, m_pelvis_q)
        rot_quat = quat_multiply(quat_conjugate(aligned_ref), r_pelvis_q)
        rot_mat = quat_to_rotmat(rot_quat).T
        obs_recon[t, 54:60] = rotmat_to_6d(rot_mat)

        # [60:63] base_ang_vel — C++ reads from IMU; we don't have it in CSV
        obs_recon[t, 60:63] = 0.0

        # [63:90] joint_pos_rel = motor_q - default_joint_pos
        obs_recon[t, 63:90] = (motor_q[t] - default_joint_pos).astype(np.float32)

        # [90:117] joint_vel_rel = motor_dq - default_joint_vel (zero vector)
        obs_recon[t, 90:117] = motor_dq[t].astype(np.float32)  # default_joint_vel is zero

        # [117:144] last_action = previous step's raw action
        if t > 0:
            obs_recon[t, 117:144] = raw_actions[t - 1].astype(np.float32)
        else:
            obs_recon[t, 117:144] = 0.0

    return obs_recon


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_obs_segments(
    cpp_obs: np.ndarray, recon_obs: np.ndarray
) -> dict:
    """Compare C++ obs vs reconstructed obs per segment."""
    per_segment = {}
    for seg_name, (start, end) in OBS_SEGMENTS.items():
        diff = cpp_obs[:, start:end] - recon_obs[:, start:end]
        per_segment[seg_name] = {
            "max_error": float(np.max(np.abs(diff))),
            "mean_error": float(np.mean(np.abs(diff))),
        }
    return per_segment


def check_action_consistency(raw_actions: np.ndarray) -> dict:
    """Check consistency: consecutive raw_actions should be smooth."""
    if raw_actions.shape[0] < 2:
        return {"max_error": 0.0, "note": "only 1 step"}
    diffs = np.abs(raw_actions[1:] - raw_actions[:-1])
    return {
        "max_error": float(np.max(diffs)),
        "mean_error": float(np.mean(diffs)),
    }


def check_processed_vs_raw(
    raw_actions: np.ndarray,
    processed_actions: np.ndarray,
    scale: np.ndarray,
    offset: np.ndarray,
) -> dict:
    """Verify processed_action = raw_action * scale + offset."""
    expected = raw_actions * scale[np.newaxis, :] + offset[np.newaxis, :]
    diff = np.abs(processed_actions - expected)
    return {
        "max_error": float(np.max(diff)),
        "mean_error": float(np.mean(diff)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze C++ Mimic dump CSV and compare against Python training-side obs assembly."
    )
    parser.add_argument(
        "--csv", default=DEFAULT_CSV,
        help=f"Path to C++ dump CSV (default: {DEFAULT_CSV})"
    )
    parser.add_argument(
        "--motion", required=True,
        help="Path to motion NPZ file"
    )
    parser.add_argument(
        "--deploy-yaml", required=True,
        help="Path to deploy.yaml"
    )
    parser.add_argument(
        "--output-dir", default=str(OUTPUT_DIR),
        help=f"Output directory for summary JSON (default: {OUTPUT_DIR})"
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    df = load_dump(args.csv)
    motion = load_motion(args.motion)
    deploy_cfg = load_deploy_yaml(args.deploy_yaml)

    default_joint_pos = deploy_cfg["default_joint_pos"]
    action_scale = deploy_cfg["action_scale"]
    action_offset = deploy_cfg["action_offset"]

    # 2. Extract arrays from CSV
    cpp_obs = extract_obs(df)
    raw_actions = extract_raw_actions(df)
    processed_actions = extract_processed_actions(df)
    motor_q = extract_motor_q(df)
    motor_dq = extract_motor_dq(df)

    total_steps = len(df)
    print(f"Total steps: {total_steps}")
    print(f"CSV obs shape: {cpp_obs.shape}")
    print(f"motor_q shape: {motor_q.shape}")
    print(f"motor_dq shape: {motor_dq.shape}")

    # 3. Reconstruct Python training-side observations
    print("\nReconstructing Python training-side observations ...")
    recon_obs = reconstruct_training_obs(
        motor_q, motor_dq, raw_actions, motion, default_joint_pos
    )

    # 4. Compare C++ obs vs reconstructed obs
    per_segment = compare_obs_segments(cpp_obs, recon_obs)

    # Overall obs error
    obs_diff = cpp_obs - recon_obs
    overall_max = float(np.max(np.abs(obs_diff)))
    overall_mean = float(np.mean(np.abs(obs_diff)))

    # 5. Action consistency checks
    action_consistency = check_action_consistency(raw_actions)
    processed_check = check_processed_vs_raw(
        raw_actions, processed_actions, action_scale, action_offset
    )

    # ---- Print summary ----
    print("\n" + "=" * 80)
    print("  C++ Mimic Dump Analysis — Observation Comparison")
    print("=" * 80)
    print(f"  CSV file     : {args.csv}")
    print(f"  Motion file  : {args.motion}")
    print(f"  Deploy YAML  : {args.deploy_yaml}")
    print(f"  Total steps  : {total_steps}")

    print(f"\n  {'Segment':<30s} {'Max Error':>16s} {'Mean Error':>16s}")
    print(f"  {'-'*30} {'-'*16} {'-'*16}")
    for seg_name, stats in per_segment.items():
        print(f"  {seg_name:<30s} {stats['max_error']:>16.9e} {stats['mean_error']:>16.9e}")

    print(f"\n  {'OVERALL':<30s} {overall_max:>16.9e} {overall_mean:>16.9e}")

    print(f"\n  Action consistency (consecutive raw_actions):")
    print(f"    Max abs delta : {action_consistency['max_error']:.9e}")
    print(f"    Mean abs delta: {action_consistency['mean_error']:.9e}")

    print(f"\n  Processed action check (raw * scale + offset):")
    print(f"    Max error : {processed_check['max_error']:.9e}")
    print(f"    Mean error: {processed_check['mean_error']:.9e}")

    # ---- Top-K error dimensions ----
    max_per_dim = np.max(np.abs(obs_diff), axis=0)
    top_idx = np.argsort(-max_per_dim)[:15]
    print(f"\n  Top-15 max-error dimensions:")
    for rank, dim in enumerate(top_idx):
        dim_max = max_per_dim[dim]
        dim_mean = float(np.mean(np.abs(obs_diff[:, dim])))
        print(f"    dim {dim:3d} | max={dim_max:.9e}  mean={dim_mean:.9e}")

    # ---- Build JSON summary ----
    summary = {
        "total_steps": total_steps,
        "csv_file": args.csv,
        "motion_file": args.motion,
        "deploy_yaml": args.deploy_yaml,
        "per_segment": per_segment,
        "overall_obs_max_error": overall_max,
        "overall_obs_mean_error": overall_mean,
        "action_max_error": action_consistency["max_error"],
        "action_mean_error": action_consistency["mean_error"],
        "processed_action_check": processed_check,
        "top_error_dims": [
            {
                "dim": int(d),
                "max_error": float(max_per_dim[d]),
                "mean_error": float(np.mean(np.abs(obs_diff[:, d]))),
            }
            for d in top_idx
        ],
    }

    json_path = out_dir / "mimic_cpp_dump_analysis_summary.json"
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nJSON summary saved to {json_path}")
    print("\n" + json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
