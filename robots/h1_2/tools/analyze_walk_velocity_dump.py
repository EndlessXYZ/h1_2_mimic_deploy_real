#!/usr/bin/env python3
"""
Analyze WalkVelocity C++ dump data.

Compares C++ deploy observations against Python training-side reconstruction.
WalkVelocity uses 47-dim observations (legs only) and 12-dim actions.

Observation layout (47 dims):
  [0:3]   base_ang_vel_pelvis   - IMU gyro in pelvis frame (scale 0.25)
  [3:6]   projected_gravity_pelvis - gravity in pelvis frame (scale 1.0)
  [6:9]   velocity_commands     - joystick * max_cmd * cmd_scale
  [9:21]  joint_pos_rel         - (q - default_q) for legs 0-11 (scale 1.0)
  [21:33] joint_vel_rel         - dq for legs 0-11 (scale 0.05)
  [33:45] last_action           - previous raw action (12 dims)
  [45:47] gait_phase            - sin/cos of phase (period 0.8s)

Usage:
  python analyze_walk_velocity_dump.py \
      --csv /tmp/wv_dump.csv \
      --deploy-yaml ../config/policy/walk_velocity/v0/params/deploy.yaml
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Constants
NUM_OBS = 47
NUM_ACTIONS = 12
NUM_MOTORS = 27
NUM_LEG_MOTORS = 12

# Observation segments
OBS_SEGMENTS = {
    "base_ang_vel":         (0, 3),
    "projected_gravity":    (3, 6),
    "velocity_commands":    (6, 9),
    "joint_pos_rel":        (9, 21),
    "joint_vel_rel":        (21, 33),
    "last_action":          (33, 45),
    "gait_phase":           (45, 47),
}


def load_dump(path: str) -> pd.DataFrame:
    """Load C++ dump CSV."""
    p = Path(path)
    if not p.exists():
        print(f"[ERROR] CSV file not found: {path}")
        sys.exit(1)
    df = pd.read_csv(p)
    print(f"Loaded {len(df)} rows from {path}")
    return df


def load_deploy_yaml(path: str) -> dict:
    """Load deploy.yaml and extract configuration."""
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

    # Observation scales
    obs_cfg = cfg.get("observations", {})
    ang_vel_scale = np.array(obs_cfg.get("base_ang_vel_pelvis", {}).get("scale", [0.25] * 3), dtype=np.float32)
    joint_vel_scale = np.array(obs_cfg.get("joint_vel_rel", {}).get("scale", [0.05] * 12), dtype=np.float32)
    cmd_scale = np.array(obs_cfg.get("velocity_commands", {}).get("scale", [2.0, 2.0, 0.25]), dtype=np.float32)

    print(f"Loaded deploy.yaml: {len(default_joint_pos)} default joints, "
          f"scale shape={scale.shape}, offset shape={offset.shape}")
    return {
        "default_joint_pos": default_joint_pos,
        "action_scale": scale,
        "action_offset": offset,
        "ang_vel_scale": ang_vel_scale,
        "joint_vel_scale": joint_vel_scale,
        "cmd_scale": cmd_scale,
    }


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


def compare_obs_segments(cpp_obs: np.ndarray, recon_obs: np.ndarray) -> dict:
    """Compare C++ obs vs reconstructed obs per segment."""
    per_segment = {}
    for seg_name, (start, end) in OBS_SEGMENTS.items():
        diff = cpp_obs[:, start:end] - recon_obs[:, start:end]
        # Handle NaN values
        valid_mask = ~(np.isnan(diff).any(axis=1))
        if valid_mask.sum() == 0:
            per_segment[seg_name] = {"max_error": float("nan"), "mean_error": float("nan")}
        else:
            valid_diff = diff[valid_mask]
            per_segment[seg_name] = {
                "max_error": float(np.max(np.abs(valid_diff))),
                "mean_error": float(np.mean(np.abs(valid_diff))),
            }
    return per_segment


def check_action_consistency(raw_actions: np.ndarray) -> dict:
    """Check consistency: consecutive raw_actions should be smooth."""
    if raw_actions.shape[0] < 2:
        return {"max_error": 0.0, "note": "only 1 step"}
    valid_mask = ~(np.isnan(raw_actions).any(axis=1))
    if valid_mask.sum() < 2:
        return {"max_error": float("nan"), "mean_error": float("nan")}
    valid_actions = raw_actions[valid_mask]
    diffs = np.abs(valid_actions[1:] - valid_actions[:-1])
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
    valid_mask = ~(np.isnan(raw_actions).any(axis=1) | np.isnan(processed_actions).any(axis=1))
    if valid_mask.sum() == 0:
        return {"max_error": float("nan"), "mean_error": float("nan")}
    valid_raw = raw_actions[valid_mask]
    valid_processed = processed_actions[valid_mask]
    expected = valid_raw * scale[np.newaxis, :] + offset[np.newaxis, :]
    diff = np.abs(valid_processed - expected)
    return {
        "max_error": float(np.max(diff)),
        "mean_error": float(np.mean(diff)),
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze WalkVelocity C++ dump")
    parser.add_argument("--csv", required=True, help="Path to dump CSV")
    parser.add_argument("--deploy-yaml", required=True, help="Path to deploy.yaml")
    parser.add_argument("--output-dir", default="/home/meme/Documents/unitree_h1/artifacts/walk_velocity_cpp_dump_analysis")
    args = parser.parse_args()

    # Load data
    df = load_dump(args.csv)
    deploy_cfg = load_deploy_yaml(args.deploy_yaml)

    # Extract arrays
    cpp_obs = extract_obs(df)
    raw_actions = extract_raw_actions(df)
    processed_actions = extract_processed_actions(df)
    motor_q = extract_motor_q(df)
    motor_dq = extract_motor_dq(df)

    T = len(df)
    print(f"Total steps: {T}")
    print(f"CSV obs shape: {cpp_obs.shape}")
    print(f"motor_q shape: {motor_q.shape}")
    print(f"motor_dq shape: {motor_dq.shape}")

    # Reconstruct Python training-side observations
    # Note: We don't have the actual IMU data in the CSV, so we can only
    # reconstruct joint_pos_rel, joint_vel_rel, and last_action accurately.
    # base_ang_vel and projected_gravity require IMU data which is not dumped.
    print("\nReconstructing Python training-side observations ...")

    default_joint_pos = deploy_cfg["default_joint_pos"]
    joint_vel_scale = deploy_cfg["joint_vel_scale"]

    recon_obs = np.zeros((T, NUM_OBS), dtype=np.float32)

    for t in range(T):
        # [0:3] base_ang_vel - cannot reconstruct without IMU data
        # C++ uses pelvis frame (with waist_yaw transform), training uses torso frame
        # This is a known difference
        recon_obs[t, 0:3] = np.nan  # placeholder

        # [3:6] projected_gravity - cannot reconstruct without IMU data
        recon_obs[t, 3:6] = np.nan  # placeholder

        # [6:9] velocity_commands - cannot reconstruct without joystick data
        recon_obs[t, 6:9] = np.nan  # placeholder

        # [9:21] joint_pos_rel = (motor_q[0:12] - default_joint_pos) * scale
        # C++ applies scale 1.0, so just (q - default)
        recon_obs[t, 9:21] = (motor_q[t, :12] - default_joint_pos)

        # [21:33] joint_vel_rel = motor_dq[0:12] * scale (0.05)
        recon_obs[t, 21:33] = motor_dq[t, :12] * joint_vel_scale

        # [33:45] last_action = previous step's raw action
        if t > 0:
            recon_obs[t, 33:45] = raw_actions[t - 1]
        else:
            recon_obs[t, 33:45] = 0.0

        # [45:47] gait_phase - cannot reconstruct without timing info
        recon_obs[t, 45:47] = np.nan  # placeholder

    # Compare
    per_segment = compare_obs_segments(cpp_obs, recon_obs)

    # Action checks
    action_consistency = check_action_consistency(raw_actions)
    processed_check = check_processed_vs_raw(
        raw_actions, processed_actions,
        deploy_cfg["action_scale"], deploy_cfg["action_offset"]
    )

    # Print results
    print("\n" + "=" * 80)
    print("  C++ WalkVelocity Dump Analysis — Observation Comparison")
    print("=" * 80)
    print(f"  CSV file     : {args.csv}")
    print(f"  Deploy YAML  : {args.deploy_yaml}")
    print(f"  Total steps  : {T}")
    print()
    print(f"  {'Segment':<30} {'Max Error':>16} {'Mean Error':>16}")
    print(f"  {'-' * 30} {'-' * 16} {'-' * 16}")
    for seg_name, stats in per_segment.items():
        max_err = stats["max_error"]
        mean_err = stats["mean_error"]
        if np.isnan(max_err):
            print(f"  {seg_name:<30} {'N/A':>16} {'N/A':>16}")
        else:
            print(f"  {seg_name:<30} {max_err:>16.6e} {mean_err:>16.6e}")

    print()
    print(f"  Action consistency (consecutive raw_actions):")
    if not np.isnan(action_consistency["max_error"]):
        print(f"    Max abs delta : {action_consistency['max_error']:.6e}")
        print(f"    Mean abs delta: {action_consistency['mean_error']:.6e}")
    else:
        print(f"    N/A (no valid data)")

    print()
    print(f"  Processed action check (raw * scale + offset):")
    if not np.isnan(processed_check["max_error"]):
        print(f"    Max error : {processed_check['max_error']:.6e}")
        print(f"    Mean error: {processed_check['mean_error']:.6e}")
    else:
        print(f"    N/A (no valid data)")

    # Top error dimensions
    print()
    print("  Top-10 max-error dimensions (excluding NaN):")
    valid_dims = []
    for dim in range(NUM_OBS):
        diff = np.abs(cpp_obs[:, dim] - recon_obs[:, dim])
        valid_mask = ~np.isnan(diff)
        if valid_mask.sum() > 0:
            max_err = float(np.max(diff[valid_mask]))
            mean_err = float(np.mean(diff[valid_mask]))
            valid_dims.append((dim, max_err, mean_err))
    valid_dims.sort(key=lambda x: x[1], reverse=True)
    for dim, max_err, mean_err in valid_dims[:10]:
        seg_name = "unknown"
        for name, (start, end) in OBS_SEGMENTS.items():
            if start <= dim < end:
                seg_name = name
                break
        print(f"    dim {dim:>3} ({seg_name:<20}) | max={max_err:.6e}  mean={mean_err:.6e}")

    # Save summary
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "total_steps": T,
        "csv_file": args.csv,
        "deploy_yaml": args.deploy_yaml,
        "per_segment": per_segment,
        "action_consistency": action_consistency,
        "processed_action_check": processed_check,
        "top_error_dims": [
            {"dim": dim, "max_error": max_err, "mean_error": mean_err}
            for dim, max_err, mean_err in valid_dims[:15]
        ],
    }
    summary_path = output_dir / "walk_velocity_cpp_dump_analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nJSON summary saved to {summary_path}")


if __name__ == "__main__":
    main()
