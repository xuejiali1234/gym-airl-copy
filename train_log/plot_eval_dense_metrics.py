import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from configs.config import Config
from envs.merging_env import MergingEnv
from utils.data_loader import MergingDataset

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False

RUN_NAME = "baseline_attn_goal_safe_branch_aux_20260418_224355"
RUN_DIR = os.path.join(SCRIPT_DIR, RUN_NAME)
DEFAULT_EVAL_CSV = os.path.join(RUN_DIR, "eval_metrics.csv")
EPS = 1e-8


def get_series(df, name):
    if name not in df.columns:
        raise KeyError(f"Missing required eval metric column: {name}")
    return pd.to_numeric(df[name], errors="coerce")


def split_dataset(dataset, train_ratio=0.8, seed=42):
    num_trajs = len(dataset.trajectories)
    if num_trajs < 2:
        return dataset, dataset

    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_trajs)
    split_idx = max(1, min(num_trajs - 1, int(num_trajs * train_ratio)))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    def make_subset(idxs):
        subset = object.__new__(dataset.__class__)
        if hasattr(dataset, "data_paths"):
            subset.data_paths = dataset.data_paths
        subset.device = getattr(dataset, "device", "cpu")
        subset.cfg = getattr(dataset, "cfg", Config())
        subset.trajectories = [dataset.trajectories[int(i)] for i in idxs]
        subset.expert_mean = dataset.expert_mean
        subset.expert_std = dataset.expert_std
        if hasattr(dataset, "confidence_weights"):
            subset.confidence_weights = np.ones(len(subset.trajectories), dtype=np.float32)
        return subset

    return make_subset(train_indices), make_subset(val_indices)


def expert_action_provider(env):
    actions = env.current_traj.get("action")
    if actions is None or len(actions) == 0:
        return np.zeros(2, dtype=np.float32)
    action_idx = min(int(env.t), len(actions) - 1)
    return np.asarray(actions[action_idx], dtype=np.float32)


def compute_expert_eval_dense_norm100(n_eval_episodes):
    cfg = Config()
    data_paths = [
        os.path.join(REPO_ROOT, "data", "lane_change_trajectories-0750am-0805am"),
        os.path.join(REPO_ROOT, "data", "lane_change_trajectories-0805am-0820am"),
        os.path.join(REPO_ROOT, "data", "lane_change_trajectories-0820am-0835am"),
    ]
    dataset = MergingDataset(data_paths, device="cpu")
    _, val_dataset = split_dataset(dataset, train_ratio=0.8, seed=cfg.SEED)
    env = MergingEnv(val_dataset)
    dense_returns = []

    for ep_idx in range(n_eval_episodes):
        obs, _ = env.reset(seed=cfg.SEED + ep_idx)
        terminated = False
        truncated = False
        dense_return = 0.0
        while not (terminated or truncated):
            action = expert_action_provider(env)
            obs, reward, terminated, truncated, info = env.step(action)
            dense_return += float(info.get("eval_dense_reward", 0.0))
        dense_returns.append(dense_return)

    return 10.0 * float(np.mean(dense_returns))


def add_epoch0_expert_normalized_score(full_df, expert_dense_norm100):
    epoch0_rows = full_df[full_df["epoch"] == 0]
    if epoch0_rows.empty:
        raise ValueError("Full-eval data must contain epoch=0 for epoch0/expert normalization.")

    epoch0_dense = float(epoch0_rows.iloc[0]["eval_dense_return_norm100"])
    denom = expert_dense_norm100 - epoch0_dense
    if abs(denom) < EPS:
        denom = EPS

    full_df = full_df.copy()
    full_df["dense_return_epoch0_baseline"] = epoch0_dense
    full_df["dense_return_expert_baseline"] = expert_dense_norm100
    full_df["dense_return_rel_epoch0_expert"] = (
        100.0 * (full_df["eval_dense_return_norm100"] - epoch0_dense) / denom
    )
    return full_df, epoch0_dense, expert_dense_norm100


def plot_eval_dense_metrics(eval_csv_path, save_dir=None):
    if save_dir is None:
        save_dir = os.path.dirname(eval_csv_path)
    df = pd.read_csv(eval_csv_path)
    full_eval_episodes = int(get_series(df, "eval_n_episodes").max())
    full_df = df[df["eval_n_episodes"] == full_eval_episodes].copy()
    if full_df.empty:
        raise ValueError("No full-eval rows found in eval_metrics.csv")

    expert_dense_norm100 = compute_expert_eval_dense_norm100(full_eval_episodes)
    full_df, epoch0_dense, expert_dense_norm100 = add_epoch0_expert_normalized_score(
        full_df,
        expert_dense_norm100,
    )
    epoch = get_series(full_df, "epoch")

    dense_rel = get_series(full_df, "dense_return_rel_epoch0_expert")
    endpoint = get_series(full_df, "endpoint_success_rate") * 100.0
    safety = get_series(full_df, "safety_success_rate") * 100.0
    collision = get_series(full_df, "collision_rate") * 100.0

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(
        epoch,
        dense_rel,
        color="#1f77b4",
        marker="o",
        linewidth=2.3,
        label="Dense Return Relative Score",
    )
    axes[0].axhline(0.0, color="#666666", linestyle="--", linewidth=1.2, label="Epoch 0 baseline")
    axes[0].axhline(100.0, color="#111111", linestyle=":", linewidth=1.2, label="Expert baseline")
    axes[0].set_title("Full-Eval Dense Policy Return", fontsize=15, fontweight="bold")
    axes[0].set_ylabel("Relative Score (epoch0=0, expert=100)", fontsize=12)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(loc="best", fontsize=10)

    axes[1].plot(epoch, endpoint, color="#ff7f0e", marker="o", linewidth=2.0, label="Endpoint Success (%)")
    axes[1].plot(epoch, safety, color="#2ca02c", marker="o", linewidth=2.0, label="Safety Success (%)")
    axes[1].plot(epoch, collision, color="#d62728", marker="o", linewidth=2.0, label="Collision Rate (%)")
    axes[1].set_title("Full-Eval Success and Collision Metrics", fontsize=15, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Rate (%)", fontsize=12)
    axes[1].set_ylim(0, 105)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(loc="best", fontsize=10)

    plt.tight_layout()
    csv_path = os.path.join(save_dir, "Eval_Dense_Policy_Return_FullEval_Normalized.csv")
    save_path = os.path.join(save_dir, "Eval_Dense_Policy_Return_FullEval_Normalized.png")
    full_df.to_csv(csv_path, index=False)
    print(
        "[*] Dense normalization baselines | "
        f"epoch0={epoch0_dense:.3f}, expert={expert_dense_norm100:.3f}, "
        f"full_eval_episodes={full_eval_episodes}"
    )
    print(f"Saved full-eval normalized table to: {csv_path}")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved eval dense metrics plot to: {save_path}")
    plt.show()


if __name__ == "__main__":
    latest_eval = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_EVAL_CSV
    if not os.path.exists(latest_eval):
        raise FileNotFoundError(
            "Set RUN_NAME at the top of this script, or pass a run eval_metrics.csv path, "
            "for example: python train_log/plot_eval_dense_metrics.py train_log/<run>/eval_metrics.csv"
        )
    plot_eval_dense_metrics(latest_eval)
