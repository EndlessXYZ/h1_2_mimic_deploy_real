#!/usr/bin/env python3
"""
Analyze C++ WalkVelocity dump CSV (/tmp/wv_dump.csv).

Compares C++ raw_action against Python ONNX inference output,
reports per-dimension / per-component error statistics,
and checks transition quality (motor discontinuity, first-step action magnitude).

Output:
  - Summary printed to stdout
  - JSON saved to artifacts/walk_velocity_rollout_h1_2/cpp_dump_analysis_summary.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import onnxruntime as ort

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_DUMP_FILE = "/tmp/wv_dump.csv"
DEFAULT_ONNX_PATH = (
    "/home/meme/Documents/unitree_h1/h1_2_mimic_deploy_real/robots/h1_2"
    "/config/policy/walk_velocity/v0/exported/policy.onnx"
)
OUTPUT_DIR = Path("/home/meme/Documents/unitree_h1/artifacts/walk_velocity_rollout_h1_2")
OUTPUT_JSON = OUTPUT_DIR / "cpp_dump_analysis_summary.json"

NUM_OBS = 47
NUM_ACTIONS = 12
NUM_MOTORS = 27

# Observation component groups (inclusive ranges)
COMPONENTS = {
    "base_ang_vel":      (0, 2),
    "projected_gravity": (3, 5),
    "velocity_commands": (6, 8),
    "joint_pos_rel":     (9, 20),
    "joint_vel_rel":     (21, 32),
    "last_action":       (33, 44),
    "gait_phase":        (45, 46),
}

TRANSITION_STEPS = 5  # number of initial steps to inspect for transition quality


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_dump(path: str) -> pd.DataFrame:
    """Load the C++ dump CSV. Exits gracefully if the file is missing."""
    p = Path(path)
    if not p.exists():
        print(f"[ERROR] Dump file not found: {path}")
        sys.exit(1)
    df = pd.read_csv(p)
    print(f"Loaded {len(df)} rows from {path}")
    return df


def extract_obs(df: pd.DataFrame) -> np.ndarray:
    """Extract (T, 47) float32 observation matrix."""
    cols = [f"obs_{i}" for i in range(NUM_OBS)]
    return df[cols].values.astype(np.float32)


def extract_cpp_raw_actions(df: pd.DataFrame) -> np.ndarray:
    """Extract (T, 12) float32 C++ raw_action matrix."""
    cols = [f"raw_action_{i}" for i in range(NUM_ACTIONS)]
    return df[cols].values.astype(np.float32)


def extract_cpp_processed_actions(df: pd.DataFrame) -> np.ndarray:
    """Extract (T, 12) float32 C++ processed_action matrix."""
    cols = [f"processed_action_{i}" for i in range(NUM_ACTIONS)]
    return df[cols].values.astype(np.float32)


def extract_motor_q(df: pd.DataFrame) -> np.ndarray:
    """Extract (T, 27) float32 motor positions."""
    cols = [f"motor_q_{i}" for i in range(NUM_MOTORS)]
    return df[cols].values.astype(np.float32)


def extract_motor_dq(df: pd.DataFrame) -> np.ndarray:
    """Extract (T, 27) float32 motor velocities."""
    cols = [f"motor_dq_{i}" for i in range(NUM_MOTORS)]
    return df[cols].values.astype(np.float32)


def run_onnx_inference(obs_seq: np.ndarray, onnx_path: str) -> np.ndarray:
    """Run Python ONNX inference on each observation row. Returns (T, 12) float32."""
    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name

    outputs = []
    for obs in obs_seq:
        act = ort_session.run(None, {input_name: obs.reshape(1, -1).astype(np.float32)})[0]
        act = act.squeeze().astype(np.float32)
        outputs.append(act)
    return np.stack(outputs, axis=0)


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def compute_action_errors(cpp_raw: np.ndarray, py_onnx: np.ndarray) -> dict:
    """Per-dimension and overall error between C++ raw_action and Python ONNX output."""
    diff = cpp_raw - py_onnx
    abs_diff = np.abs(diff)

    per_dim_max = np.max(abs_diff, axis=0)
    per_dim_mean = np.mean(abs_diff, axis=0)

    overall_max = float(np.max(per_dim_max))
    overall_mean = float(np.mean(per_dim_mean))

    per_dim = []
    for d in range(NUM_ACTIONS):
        per_dim.append({
            "dim": d,
            "max_abs_error": float(per_dim_max[d]),
            "mean_abs_error": float(per_dim_mean[d]),
        })

    return {
        "overall_max_abs_error": overall_max,
        "overall_mean_abs_error": overall_mean,
        "per_dimension": per_dim,
    }


def compute_component_errors(cpp_raw: np.ndarray, py_onnx: np.ndarray) -> dict:
    """Group action dimensions into observation components and report errors.

    Since raw_action is 12-dim (one per leg joint), we map action dims to the
    joint-related observation components for context. The action itself is not
    split by component, so we report the full 12-dim error but also show the
    observation component statistics for reference.
    """
    # We report action error grouped by approximate joint mapping:
    # Actions 0-5: left leg (hip_yaw, hip_roll, hip_pitch, knee, ankle_pitch, ankle_roll)
    # Actions 6-11: right leg (same ordering)
    action_groups = {
        "left_leg (0-5)":  (0, 5),
        "right_leg (6-11)": (6, 11),
    }
    diff = np.abs(cpp_raw - py_onnx)
    result = {}
    for name, (s, e) in action_groups.items():
        group_diff = diff[:, s:e + 1]
        result[name] = {
            "max_abs_error": float(np.max(group_diff)),
            "mean_abs_error": float(np.mean(group_diff)),
        }
    return result


def compute_obs_component_stats(obs: np.ndarray) -> dict:
    """Report observation value statistics per component (first row + overall)."""
    stats = {}
    for name, (s, e) in COMPONENTS.items():
        comp = obs[:, s:e + 1]
        stats[name] = {
            "first_step_values": comp[0].tolist(),
            "overall_min": float(np.min(comp)),
            "overall_max": float(np.max(comp)),
            "overall_mean": float(np.mean(comp)),
        }
    return stats


def analyze_transition_quality(df: pd.DataFrame, motor_q: np.ndarray,
                               cpp_raw: np.ndarray, processed: np.ndarray) -> dict:
    """Check motor position discontinuity and first-step action magnitude."""
    n_steps = min(TRANSITION_STEPS, len(df))

    # Motor position discontinuity: |q_t - q_{t-1}| for first few steps
    motor_discontinuity = []
    for t in range(1, n_steps):
        delta = np.abs(motor_q[t] - motor_q[t - 1])
        motor_discontinuity.append({
            "step": int(df.iloc[t].get("step", t)),
            "max_abs_delta": float(np.max(delta)),
            "mean_abs_delta": float(np.mean(delta)),
            "per_joint_max": delta.tolist(),
        })

    # First-step action magnitude
    first_raw = cpp_raw[0]
    first_processed = processed[0]

    return {
        "num_transition_steps_inspected": n_steps,
        "motor_position_discontinuity": motor_discontinuity,
        "first_step_raw_action": {
            "values": first_raw.tolist(),
            "max_abs": float(np.max(np.abs(first_raw))),
            "mean_abs": float(np.mean(np.abs(first_raw))),
        },
        "first_step_processed_action": {
            "values": first_processed.tolist(),
            "max_abs": float(np.max(np.abs(first_processed))),
            "mean_abs": float(np.mean(np.abs(first_processed))),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze C++ WalkVelocity dump CSV and compare with Python ONNX inference."
    )
    parser.add_argument(
        "--dump-file", default=DEFAULT_DUMP_FILE,
        help=f"Path to the C++ dump CSV (default: {DEFAULT_DUMP_FILE})"
    )
    parser.add_argument(
        "--onnx-path", default=DEFAULT_ONNX_PATH,
        help=f"Path to the ONNX policy model (default: {DEFAULT_ONNX_PATH})"
    )
    args = parser.parse_args()

    # 1. Load CSV
    df = load_dump(args.dump_file)
    total_steps = len(df)

    # 2. Extract arrays
    obs = extract_obs(df)
    cpp_raw = extract_cpp_raw_actions(df)
    cpp_processed = extract_cpp_processed_actions(df)
    motor_q = extract_motor_q(df)
    motor_dq = extract_motor_dq(df)

    # 3. ONNX inference
    print(f"Running ONNX inference on {total_steps} steps ...")
    py_onnx = run_onnx_inference(obs, args.onnx_path)
    print("ONNX inference complete.")

    # 4. Action error statistics
    action_errors = compute_action_errors(cpp_raw, py_onnx)
    component_errors = compute_component_errors(cpp_raw, py_onnx)

    # 5. Observation component stats
    obs_stats = compute_obs_component_stats(obs)

    # 6. Transition quality
    transition = analyze_transition_quality(df, motor_q, cpp_raw, cpp_processed)

    # ---- Print summary ----
    print("\n" + "=" * 70)
    print("C++ WalkVelocity Dump Analysis")
    print("=" * 70)
    print(f"  Dump file : {args.dump_file}")
    print(f"  ONNX model: {args.onnx_path}")
    print(f"  Total steps: {total_steps}")

    print(f"\n--- C++ raw_action vs Python ONNX output ---")
    print(f"  Overall max abs error : {action_errors['overall_max_abs_error']:.6e}")
    print(f"  Overall mean abs error: {action_errors['overall_mean_abs_error']:.6e}")
    print(f"  Per-dimension breakdown:")
    for d in action_errors["per_dimension"]:
        print(f"    dim {d['dim']:2d}: max={d['max_abs_error']:.6e}  mean={d['mean_abs_error']:.6e}")

    print(f"\n--- Action error by component group ---")
    for name, vals in component_errors.items():
        print(f"  {name:25s}  max={vals['max_abs_error']:.6e}  mean={vals['mean_abs_error']:.6e}")

    print(f"\n--- Observation component statistics ---")
    for name, vals in obs_stats.items():
        print(f"  {name:25s}  min={vals['overall_min']:.4f}  max={vals['overall_max']:.4f}  "
              f"mean={vals['overall_mean']:.4f}")

    print(f"\n--- Transition quality (first {TRANSITION_STEPS} steps) ---")
    print(f"  First-step raw action      : max_abs={transition['first_step_raw_action']['max_abs']:.4f}  "
          f"mean_abs={transition['first_step_raw_action']['mean_abs']:.4f}")
    print(f"  First-step processed action: max_abs={transition['first_step_processed_action']['max_abs']:.4f}  "
          f"mean_abs={transition['first_step_processed_action']['mean_abs']:.4f}")
    if transition["motor_position_discontinuity"]:
        print(f"  Motor position discontinuity (|q_t - q_{{t-1}}|):")
        for entry in transition["motor_position_discontinuity"]:
            print(f"    step {entry['step']}: max={entry['max_abs_delta']:.6f}  "
                  f"mean={entry['mean_abs_delta']:.6f}")
    else:
        print("  (only 1 step in dump — cannot compute motor discontinuity)")

    # ---- Build JSON summary ----
    summary = {
        "dump_file": args.dump_file,
        "onnx_path": args.onnx_path,
        "total_steps": total_steps,
        "action_errors": action_errors,
        "action_component_errors": component_errors,
        "observation_component_stats": obs_stats,
        "transition_quality": transition,
    }

    # ---- Save JSON ----
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nJSON summary saved to {OUTPUT_JSON}")

    print("\n" + json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
