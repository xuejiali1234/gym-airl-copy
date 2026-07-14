import os
import random
import sys
import traceback
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection
from stable_baselines3 import PPO

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from configs.config import Config
from envs.merging_env import MergingEnv
from model.attention_net import AttentionFeaturesExtractor, GoalConditionedMLPFeaturesExtractor
from utils.data_loader import MergingDataset


DEFAULT_MODEL_TAG = "PPO_RL_Goal_PushMerge_E300_epoch290"
DEFAULT_MODEL_PATH = os.path.join(
    root_dir,
    "train_log",
    "ppo_rl_baseline_PPO_RL_Goal_PushMerge_E300_20260622_122052",
    "checkpoints",
    "ppo_rl_policy_PPO_RL_Goal_PushMerge_E300_epoch_290.zip",
)
MODEL_TAG = os.environ.get("RL_VIS_MODEL_TAG", DEFAULT_MODEL_TAG).strip() or DEFAULT_MODEL_TAG
MODEL_PATH = os.environ.get("RL_VIS_MODEL_PATH", DEFAULT_MODEL_PATH).strip() or DEFAULT_MODEL_PATH

FT_TO_M = 0.3048
DEFAULT_NUM_SAMPLES = 20
DEFAULT_RANDOM_SEED = 42

matplotlib.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.size": 10,
        "axes.unicode_minus": False,
        "axes.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "legend.frameon": False,
    }
)


class SingleTrajDataset:
    def __init__(self, traj, expert_mean, expert_std):
        self.trajectories = [traj]
        self.expert_mean = expert_mean
        self.expert_std = expert_std

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.trajectories[idx]


def build_rl_cfg():
    cfg = Config()
    cfg.ENABLE_GOAL_CONDITION = True
    cfg.GOAL_ABLATION_MODE = "normal"
    cfg.ENABLE_ATTENTION = False
    cfg.ATTENTION_ABLATION_MODE = "normal"
    cfg.ENABLE_SAFETY_MODULE = False
    cfg.ENABLE_SAFETY_BRANCH = False
    cfg.ENABLE_SAFETY_AUX_LOSS = False
    cfg.ENABLE_PREDICTIVE_SAFETY_CRITIC = False
    cfg.ENABLE_PREDICTIVE_SAFETY_RESIDUAL = False
    cfg.PREDICTIVE_SAFETY_ENABLE_CPAIR_ADDITIVE = False
    return cfg


def run_inference(model, env):
    model_x, model_y, model_v = [], [], []
    rollout_rows = []

    obs, _ = env.reset()
    px, py, vx, vy = env.ego_state
    model_x.append(px)
    model_y.append(py)
    model_v.append(np.sqrt(vx ** 2 + vy ** 2))

    done = False
    truncated = False
    max_steps = len(env.current_traj["ego_pos"]) + 50
    step_count = 0

    while not (done or truncated) and step_count < max_steps:
        prev_px, prev_py, prev_vx, prev_vy = env.ego_state.copy()
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, truncated, info = env.step(action)

        px, py, vx, vy = env.ego_state
        model_x.append(px)
        model_y.append(py)
        model_v.append(np.sqrt(vx ** 2 + vy ** 2))
        rollout_rows.append(
            {
                "step": step_count,
                "pred_px_ft": float(px),
                "pred_py_ft": float(py),
                "pred_vx_ftps": float(vx),
                "pred_vy_ftps": float(vy),
                "pred_speed_ftps": float(np.sqrt(vx ** 2 + vy ** 2)),
                "pred_ax_cmd_ftps2": float(action[0] * env.cfg.PHYS_STEER_MAX),
                "pred_ay_cmd_ftps2": float(action[1] * env.cfg.PHYS_ACC_MAX),
                "pred_dx_ft": float(px - prev_px),
                "pred_dy_ft": float(py - prev_py),
                "pred_min_ttc_s": float(info.get("eval_min_ttc", 20.0)),
                "pred_min_thw_s": float(info.get("eval_min_thw", 10.0)),
                "pred_merge_success": bool(info.get("is_merge_success", False)),
                "pred_endpoint_success": bool(info.get("is_endpoint_success", False)),
                "pred_safety_success": bool(info.get("is_safety_success", False)),
                "pred_collided": bool(info.get("is_collided", False)),
            }
        )
        step_count += 1

    return model_x, model_y, model_v, rollout_rows


def resolve_sample_indices(global_dataset):
    target_filename = os.environ.get("VIS_TARGET_FILENAME", "").strip()
    if target_filename:
        target_basename = os.path.basename(target_filename)
        matched = [
            idx
            for idx in range(len(global_dataset))
            if os.path.basename(str(global_dataset[idx].get("filename", ""))) == target_basename
        ]
        if not matched:
            matched = [
                idx
                for idx in range(len(global_dataset))
                if target_basename in os.path.basename(str(global_dataset[idx].get("filename", "")))
            ]
        if not matched:
            raise FileNotFoundError(f"Cannot find target trajectory: {target_filename}")
        return matched[:1]

    raw_num_samples = os.environ.get("VIS_NUM_SAMPLES", str(DEFAULT_NUM_SAMPLES)).strip()
    if raw_num_samples.lower() in {"all", "*", "0"}:
        return list(range(len(global_dataset)))

    num_samples = min(int(raw_num_samples), len(global_dataset))
    rng = random.Random(int(os.environ.get("VIS_RANDOM_SEED", str(DEFAULT_RANDOM_SEED))))
    return rng.sample(range(len(global_dataset)), num_samples)


def draw_single_trajectory(cfg, traj, model, output_dir, index):
    filename = traj.get("filename", f"trajectory_{index}.csv")
    single_dataset = SingleTrajDataset(traj, traj["expert_mean"], traj["expert_std"])
    env = MergingEnv(single_dataset, cfg=cfg)

    gt_x = traj["ego_pos"][:, 0]
    gt_y = traj["ego_pos"][:, 1]
    gt_vx = traj["ego_vel"][:, 0]
    gt_vy = traj["ego_vel"][:, 1]
    gt_v = np.sqrt(gt_vx ** 2 + gt_vy ** 2)

    model_x, model_y, model_v, rollout_rows = run_inference(model, env)

    gt_x_m = gt_x * FT_TO_M
    gt_y_m = gt_y * FT_TO_M
    gt_v_mps = gt_v * FT_TO_M
    model_x_m = np.asarray(model_x) * FT_TO_M
    model_y_m = np.asarray(model_y) * FT_TO_M
    model_v_mps = np.asarray(model_v) * FT_TO_M

    fig, ax = plt.subplots(figsize=(5.1, 5.0))

    vmin_val, vmax_val = 0.0, 25.0
    y_min = min(np.min(gt_y_m), np.min(model_y_m)) - 6.0
    y_max = max(np.max(gt_y_m), np.max(model_y_m)) + 6.0
    road_x_min = cfg.X_MIN * FT_TO_M
    road_x_max = cfg.X_MAX * FT_TO_M
    lane_divider_x = (cfg.X_MIN + cfg.LANE_WIDTH) * FT_TO_M

    ax.vlines(
        lane_divider_x,
        y_min,
        y_max,
        color="0.15",
        linestyle="--",
        linewidth=1.0,
        alpha=0.85,
        zorder=1,
    )

    gt_points = np.array([gt_x_m, gt_y_m]).T.reshape(-1, 1, 2)
    gt_segments = np.concatenate([gt_points[:-1], gt_points[1:]], axis=1)
    gt_collection = LineCollection(gt_segments, cmap="RdYlGn_r", linewidth=2.2, alpha=0.75, zorder=2)
    gt_collection.set_array(gt_v_mps)
    gt_collection.set_clim(vmin=vmin_val, vmax=vmax_val)
    ax.add_collection(gt_collection)
    ax.plot(
        gt_x_m,
        gt_y_m,
        color="black",
        linewidth=1.2,
        alpha=0.65,
        label="Truth",
        zorder=3,
    )

    pred_points = ax.scatter(
        model_x_m,
        model_y_m,
        c=model_v_mps,
        cmap="RdYlGn_r",
        s=13,
        edgecolors="black",
        linewidths=0.35,
        label="Policy",
        zorder=4,
        vmin=vmin_val,
        vmax=vmax_val,
    )

    ax.set_aspect(1.0 / 3.0)
    ax.set_anchor("W")
    ax.set_xlim(road_x_min, road_x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(labelsize=9, top=False, right=False)
    ax.grid(False)

    cbar = plt.colorbar(pred_points, ax=ax, fraction=0.045, pad=0.12)
    cbar.ax.set_title("Speed (m/s)", pad=8, fontsize=10)
    cbar.ax.tick_params(labelsize=9, pad=4)

    valid_xticks = [tick for tick in ax.get_xticks() if road_x_min <= tick <= road_x_max]
    right_xtick = valid_xticks[-1] if valid_xticks else road_x_max
    ax.annotate(
        "(m)",
        xy=(right_xtick, 0.0),
        xycoords=ax.get_xaxis_transform(),
        xytext=(-1, -2),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=10,
        annotation_clip=False,
    )
    ax.text(
        -0.18,
        0.965,
        "(m)",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
    )

    save_stem = os.path.splitext(os.path.basename(filename))[0]
    save_path = os.path.join(output_dir, f"{index:03d}_{save_stem}.png")
    csv_path = os.path.join(output_dir, f"{index:03d}_{save_stem}.csv")

    gt_rows = pd.DataFrame(
        {
            "step": np.arange(len(gt_x), dtype=int),
            "gt_px_ft": gt_x.astype(float),
            "gt_py_ft": gt_y.astype(float),
            "gt_vx_ftps": gt_vx.astype(float),
            "gt_vy_ftps": gt_vy.astype(float),
            "gt_speed_ftps": gt_v.astype(float),
        }
    )
    pred_rows = pd.DataFrame(rollout_rows)
    export_df = pd.merge(gt_rows, pred_rows, on="step", how="outer")

    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    export_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return save_path, csv_path


def visualize_trajectory_batch_rl():
    print("=" * 80)
    print("Start RL batch trajectory visualization")
    print("=" * 80)

    cfg = build_rl_cfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stats_data_paths = [
        os.path.join(root_dir, "data", "lane_change_trajectories-0750am-0805am"),
        os.path.join(root_dir, "data", "lane_change_trajectories-0805am-0820am"),
        os.path.join(root_dir, "data", "lane_change_trajectories-0820am-0835am"),
    ]

    try:
        global_dataset = MergingDataset(stats_data_paths, device=device)
        if len(global_dataset) == 0:
            raise ValueError("Dataset is empty.")
        expert_mean, expert_std = global_dataset.get_stats()
        print(f"[OK] Loaded dataset with {len(global_dataset)} trajectories.")
    except Exception as exc:
        print(f"[ERROR] Failed to load dataset: {exc}")
        traceback.print_exc()
        return

    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")
        print(f"[INFO] Model tag: {MODEL_TAG}")
        print(f"[INFO] Model path: {MODEL_PATH}")
        _ = AttentionFeaturesExtractor
        _ = GoalConditionedMLPFeaturesExtractor
        model = PPO.load(MODEL_PATH, device=device)
        print("[OK] RL model loaded.")
    except Exception as exc:
        print(f"[ERROR] Failed to load model: {exc}")
        traceback.print_exc()
        return

    try:
        selected_indices = resolve_sample_indices(global_dataset)
    except Exception as exc:
        print(f"[ERROR] Failed to select trajectories: {exc}")
        return

    num_samples = len(selected_indices)
    output_dir = os.path.join(root_dir, "plot", "batch_results_rl", MODEL_TAG)
    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Number of trajectories to draw: {num_samples}")
    print(f"[OUTPUT] Save dir: {output_dir}")

    for draw_idx, dataset_idx in enumerate(selected_indices, start=1):
        traj = global_dataset[dataset_idx]
        traj["expert_mean"] = expert_mean
        traj["expert_std"] = expert_std
        filename = traj.get("filename", f"trajectory_{dataset_idx}.csv")
        print(f"[{draw_idx}/{num_samples}] {filename}")

        try:
            save_path, csv_path = draw_single_trajectory(cfg, traj, model, output_dir, draw_idx)
            print(f"  -> saved {save_path}")
            print(f"  -> saved {csv_path}")
        except Exception as exc:
            print(f"[ERROR] Failed on {filename}: {exc}")
            traceback.print_exc()

    print("[OK] RL trajectory batch visualization complete.")


if __name__ == "__main__":
    visualize_trajectory_batch_rl()
