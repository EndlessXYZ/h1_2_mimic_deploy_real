#!/usr/bin/env python3
"""Mimic Golden Trace v2 - play.py aligned pipeline.

v1 manually rebuilds a MuJoCo sim and hand-writes the 150-dim obs formula,
which is a different code path from scripts/play.py. v2 reuses the same
ManagerBasedRlEnv + ObservationManager pipeline as play.py, so the golden
obs is the genuine training-side obs_manager.compute() output.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort

# Import tasks to register them in the registry.
import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
DEFAULT_ONNX = Path(
    "/home/meme/Documents/unitree_h1/h1_2_mimic_deploy_real/robots/h1_2/"
    "config/policy/mimic/v0/exported/policy.onnx")
DEFAULT_MOTION = Path(
    "/home/meme/Documents/unitree_h1/h1_2_mimic_deploy_real/robots/h1_2/"
    "config/policy/mimic/v0/params/dance1_subject2.npz")
DEFAULT_OUT_DIR = Path("/home/meme/Documents/unitree_h1/artifacts/mimic_openloop_v2")
DEFAULT_TASK = "Unitree-H1_2-Tracking-No-State-Estimation"
TORSO_JOINT_IDX = 12

# 144-dim actor obs term segments (No-State-Estimation, matches tracking_env_cfg).
OBS_SEGMENTS = [
    ("command",            0,  54),
    ("motion_anchor_ori_b", 54, 60),
    ("base_ang_vel",       60, 63),
    ("joint_pos",          63, 90),
    ("joint_vel",          90, 117),
    ("actions",           117, 144),
]


# ---------------------------------------------------------------------------
# OnnxPolicy: light wrapper matching play.py's _OnnxPolicy (no time_step).
# ---------------------------------------------------------------------------
class OnnxPolicy:

    def __init__(self, onnx_file):
        self.session = ort.InferenceSession(onnx_file)
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, obs_np):
        feed = {self.input_name: obs_np.astype(np.float32)}
        outs = self.session.run(None, feed)
        return outs[0].squeeze(0).astype(np.float32)


# ---------------------------------------------------------------------------
# Helpers: extract breakpoint data from the ManagerBasedRlEnv.
# ---------------------------------------------------------------------------

def extract_input_from_env(env):
    """Input breakpoint: raw sensor readings from MuJoCo sim data."""
    sim = env.unwrapped.sim
    mj = sim.mj_model
    data = sim.data
    # mjlab wraps data in torch tensors of shape (num_envs, ...).
    # Note: this MuJoCo model has no "imu_quat" sensor — orientation is only
    # available via the floating-base quaternion in data.qpos[:, 3:7].
    # Gyro (imu_ang_vel) is sensor[0] at adr=0, dim=3.
    sd = data.sensordata  # shape: (num_envs, 16)
    raw_imu_quat = data.qpos[0, 3:7].cpu().numpy().astype(np.float32)  # (w,x,y,z)
    raw_imu_gyro = sd[0, 0:3].cpu().numpy().astype(np.float32)        # (x,y,z)
    # Motor positions/velocities from qpos/qvel (floating base: 7 pos + 6 vel).
    n_joints = mj.nq - 7  # subtract floating base (7 = 3 pos + 4 quat)
    raw_motor_q = data.qpos[0, 7:7 + n_joints].cpu().numpy().astype(np.float32)
    raw_motor_dq = data.qvel[0, 6:6 + n_joints].cpu().numpy().astype(np.float32)
    raw_waist_yaw = np.float32(raw_motor_q[TORSO_JOINT_IDX])
    raw_waist_yaw_omega = np.float32(raw_motor_dq[TORSO_JOINT_IDX])
    return (raw_imu_quat, raw_imu_gyro, raw_motor_q, raw_motor_dq,
            raw_waist_yaw, raw_waist_yaw_omega)


def extract_motion_window_pos(env):
    """Motion window position (0-6) from command_manager time_steps."""
    motion_cmd = env.unwrapped.command_manager.get_term("motion")
    return int(motion_cmd.time_steps.cpu().numpy().item())


def extract_processed_action(env):
    """Processed action from the joint_pos action term (raw * scale + offset)."""
    term = env.unwrapped.action_manager.get_term("joint_pos")
    return term._processed_actions[0].detach().cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Mimic Golden Trace v2")
    parser.add_argument("--task", default=DEFAULT_TASK,
                        help="Task ID for load_env_cfg")
    parser.add_argument("--onnx", default=str(DEFAULT_ONNX),
                        help="Path to policy.onnx")
    parser.add_argument("--motion-file", default=str(DEFAULT_MOTION),
                        help="Path to motion NPZ file")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                        help="Output directory")
    parser.add_argument("--steps", type=int, default=500,
                        help="Number of env steps to capture")
    parser.add_argument("--device", default="cpu",
                        help="Torch device")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load env config in play mode (no corruption, no RSI). ----
    env_cfg = load_env_cfg(args.task, play=True)
    # Disable all early terminations for numerical validation.
    # play=True only sets episode_length_s = 1e9, keeping anchor_pos/ori/ee_body_pos active.
    # Those cause the env to reset every ~5 steps, breaking motion_frame continuity.
    # For validation we need a clean 500-step trajectory with linear motion frame progression.
    env_cfg.terminations = {}
    rl_cfg = load_rl_cfg(args.task)

    motion_cmd_cfg = env_cfg.commands["motion"]
    assert isinstance(motion_cmd_cfg, MotionCommandCfg), \
        f"Expected MotionCommandCfg, got {type(motion_cmd_cfg)}"
    motion_cmd_cfg.motion_file = args.motion_file

    # ---- 2. Create env + wrapper (same pipeline as scripts/play.py). ----
    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)

    # ---- 3. Create ONNX policy. ----
    policy = OnnxPolicy(args.onnx)

    # ---- 4. Allocate recording buffers. ----
    buf = {
        'raw_imu_quat': [], 'raw_imu_gyro': [],
        'raw_motor_q': [], 'raw_motor_dq': [],
        'raw_waist_yaw': [], 'raw_waist_yaw_omega': [],
        'motion_window_pos': [],
        # Per-term observation slices.
        'obs_command': [], 'obs_motion_anchor_ori_b': [],
        'obs_base_ang_vel': [], 'obs_joint_pos': [],
        'obs_joint_vel': [], 'obs_actions': [],
        # Full 144-dim observation (B and C are identical when all scale=null).
        'obs_raw': [],
        # D and F breakpoints.
        'nn_output': [], 'processed_action': [],
    }

    # ---- 5. Reset env (motion command starts at frame ~0). ----
    obs, _ = env.reset()
    print(f"Collecting {args.steps} golden trace steps ...")
    print(f"  env.step_dt = {env.unwrapped.step_dt:.4f}s  "
          f"(expect 0.020s = 50Hz)")

    # ---- 6. Main collection loop. ----
    for step_i in range(args.steps):
        # Observation from the previous env.step (RslRlVecEnvWrapper wraps it
        # as obs["actor"] with shape (1, 144)).
        obs_np = obs["actor"].detach().cpu().numpy().astype(np.float32)
        obs_1d = obs_np[0]  # squeeze batch dim → (144,)

        # --- Input breakpoint ---
        (raw_imu_quat, raw_imu_gyro, raw_motor_q, raw_motor_dq,
         raw_waist_yaw, raw_waist_yaw_omega) = extract_input_from_env(env)
        motion_window_pos = extract_motion_window_pos(env)

        # --- D breakpoint: ONNX inference ---
        nn_output = policy(obs_1d.reshape(1, -1))

        # --- F breakpoint: processed action ---
        # Drive action_manager.process_action with nn_output, then read
        # the processed (scaled+offset) result from the action term.
        action_tensor = torch.from_numpy(nn_output).unsqueeze(0).to(args.device)
        env.unwrapped.action_manager.process_action(action_tensor)
        processed_action = extract_processed_action(env)

        # --- Save all breakpoint data ---
        buf['raw_imu_quat'].append(raw_imu_quat)
        buf['raw_imu_gyro'].append(raw_imu_gyro)
        buf['raw_motor_q'].append(raw_motor_q)
        buf['raw_motor_dq'].append(raw_motor_dq)
        buf['raw_waist_yaw'].append(raw_waist_yaw)
        buf['raw_waist_yaw_omega'].append(raw_waist_yaw_omega)
        buf['motion_window_pos'].append(motion_window_pos)

        for seg_name, start, end in OBS_SEGMENTS:
            buf[f'obs_{seg_name}'].append(obs_1d[start:end])
        buf['obs_raw'].append(obs_1d)
        buf['nn_output'].append(nn_output)
        buf['processed_action'].append(processed_action)

        # --- Step env (action_manager.apply_action + physics step) ---
        action_step = torch.from_numpy(nn_output).unsqueeze(0).to(args.device)
        obs, reward, done, info = env.step(action_step)  # noqa: F841

        if step_i % 50 == 0:
            print(f"  Step {step_i:3d}/{args.steps}  "
                  f"window_pos={motion_window_pos}")

    # ---- 7. Save golden trace NPZ. ----
    data = {k: np.array(v, dtype=np.float32) for k, v in buf.items()}
    data['obs_scaled'] = data['obs_raw']  # compatibility alias (Mimic scale=null)

    # Save action scale/offset from the JointPositionAction term.
    action_term = env.unwrapped.action_manager.get_term("joint_pos")
    if isinstance(action_term.scale, torch.Tensor):
        data['action_scale'] = action_term.scale[0].cpu().numpy().astype(np.float32)
    else:
        data['action_scale'] = np.array([action_term.scale], dtype=np.float32)
    if isinstance(action_term.offset, torch.Tensor):
        data['action_offset'] = action_term.offset[0].cpu().numpy().astype(np.float32)
    else:
        data['action_offset'] = np.array([action_term.offset], dtype=np.float32)
    data['default_joint_pos'] = data['action_offset']

    npz_path = out_dir / "golden_trace.npz"
    np.savez_compressed(npz_path, **data)

    # ---- 8. Write summary JSON. ----
    summary = {
        "steps": args.steps,
        "task": args.task,
        "policy": args.onnx,
        "motion_file": args.motion_file,
        "step_dt": float(env.unwrapped.step_dt),
        "output": str(npz_path),
        "version": "v2 (play.py aligned)",
        "obs_segments": [
            {"name": n, "start": s, "end": e}
            for n, s, e in OBS_SEGMENTS
        ],
    }
    summary_path = out_dir / "golden_trace_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"[OK] Golden trace v2 saved to {npz_path}")

    env.close()


if __name__ == "__main__":
    main()
