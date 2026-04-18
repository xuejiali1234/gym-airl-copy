import glob
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from configs.config import Config
from envs.merging_env import MergingEnv
from utils.data_loader import MergingDataset


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False

RUN_NAME = "baseline_attn_goal_safe_branch_aux_20260418_180202"
RUN_DIR = os.path.join(SCRIPT_DIR, RUN_NAME)
RUN_LABEL = re.sub(r"_\d{8}_\d{6}$", "", RUN_NAME)
RUN_LABEL = re.sub(r"^baseline_", "", RUN_LABEL)
CHECKPOINT_GLOB = os.path.join(REPO_ROOT, "checkpoints", f"baseline_policy_{RUN_LABEL}_epoch_*.zip")
OUTPUT_CSV = os.path.join(RUN_DIR, f"Dense_Eval_Score_{RUN_LABEL}.csv")
OUTPUT_FIG = os.path.join(RUN_DIR, f"Dense_Eval_Score_{RUN_LABEL}.png")

N_EVAL_EPISODES = 20
SMOOTH_WINDOW = 3
SHADE_ALPHA = 0.16
EPS = 1e-8


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


def checkpoint_epoch(path):
    match = re.search(r"_epoch_(\d+)\.zip$", os.path.basename(path))
    if match is None:
        return None
    return int(match.group(1))


def compute_min_ttc(env):
    px, py, vx, vy = env.ego_state
    surr_now = env._get_surround_at_t(env.t)
    ttcs = []
    lane_x_tolerance = 10.0

    for idx in (0, 4):
        tx = surr_now[idx]
        ty = surr_now[idx + 1]
        tvy = surr_now[idx + 2]
        if tx == 0 and ty == 0:
            continue

        dx = tx - px
        dy = ty - py
        if abs(dx) > lane_x_tolerance or dy <= 0.1:
            continue

        closing_speed = vy - tvy
        if closing_speed <= 0.1:
            continue

        ttcs.append(dy / closing_speed)

    return min(ttcs) if ttcs else 20.0


def dense_eval_step_reward(env, action, prev_y, ttc):
    """Dense evaluation-only reward from scheme 2; never used for training."""
    cfg = env.cfg
    px, py, vx, vy = env.ego_state
    expert_start_y = float(env.current_traj["ego_pos"][0][1])
    expert_end_y = float(env.current_traj["ego_pos"][-1][1])
    progress_budget = max(abs(expert_end_y - expert_start_y), 1e-6)

    delta_y = max(0.0, float(py - prev_y))
    progress_term = 20.0 * delta_y / progress_budget

    # In this env, action[0] controls lateral x acceleration and action[1] controls longitudinal y acceleration.
    ax_norm = abs(float(action[0]))
    ay_norm = abs(float(action[1]))
    smooth_penalty = 0.05 * ay_norm + 0.02 * ax_norm
    risk_penalty = 0.10 * max(0.0, 1.0 - float(ttc) / 5.0)

    return progress_term - smooth_penalty - risk_penalty


def dense_eval_terminal_bonus(info):
    bonus = 0.0
    if info.get("is_merge_success", False):
        bonus += 20.0
    if info.get("is_endpoint_success", False):
        bonus += 20.0
    if info.get("is_safety_success", False):
        bonus += 40.0
    if info.get("is_collided", False):
        bonus -= 40.0
    return bonus


class RandomPolicy:
    """Uniform random policy used only as the dense-score reference baseline."""

    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)

    def predict(self, obs, deterministic=True):
        action = self.rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
        return action, None


def expert_action_provider(env, obs, ep_idx):
    actions = env.current_traj.get("action")
    if actions is None or len(actions) == 0:
        return np.zeros(2, dtype=np.float32)
    action_idx = min(int(env.t), len(actions) - 1)
    return np.asarray(actions[action_idx], dtype=np.float32)


def evaluate_dense_score(model, dataset, cfg, n_eval_episodes=N_EVAL_EPISODES, action_provider=None):
    env = MergingEnv(dataset)
    dense_scores = []
    env_returns = []
    merge_successes = []
    endpoint_successes = []
    safety_successes = []
    min_ttcs = []
    mean_abs_actions = []

    for ep_idx in range(n_eval_episodes):
        obs, _ = env.reset(seed=cfg.SEED + ep_idx)
        terminated = False
        truncated = False
        info = {}
        dense_score = 0.0
        env_return = 0.0
        ep_ttcs = []
        ep_abs_actions = []

        while not (terminated or truncated):
            prev_y = float(env.ego_state[1])
            if action_provider is not None:
                action = action_provider(env, obs, ep_idx)
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ttc = compute_min_ttc(env)

            dense_score += dense_eval_step_reward(env, action, prev_y, ttc)
            env_return += float(reward)
            ep_ttcs.append(float(ttc))
            ep_abs_actions.append(float(np.mean(np.abs(action))))

        dense_score += dense_eval_terminal_bonus(info)

        dense_scores.append(dense_score)
        env_returns.append(env_return)
        merge_successes.append(float(info.get("is_merge_success", False)))
        endpoint_successes.append(float(info.get("is_endpoint_success", False)))
        safety_successes.append(float(info.get("is_safety_success", False)))
        min_ttcs.append(float(np.min(ep_ttcs)) if ep_ttcs else 20.0)
        mean_abs_actions.append(float(np.mean(ep_abs_actions)) if ep_abs_actions else 0.0)

    return {
        "dense_eval_score_mean": float(np.mean(dense_scores)),
        "dense_eval_score_std": float(np.std(dense_scores)),
        "env_return_mean": float(np.mean(env_returns)),
        "merge_success_rate": float(np.mean(merge_successes)),
        "endpoint_success_rate": float(np.mean(endpoint_successes)),
        "safety_success_rate": float(np.mean(safety_successes)),
        "min_ttc_mean": float(np.mean(min_ttcs)),
        "mean_abs_action": float(np.mean(mean_abs_actions)),
    }


def smooth(series, window=SMOOTH_WINDOW):
    return series.rolling(window=window, min_periods=1, center=True).mean()


def add_random_normalized_score(df, random_mean, expert_mean):
    raw_denom = float(expert_mean) - float(random_mean)
    denom = raw_denom if abs(raw_denom) > EPS else EPS
    df = df.copy()
    df["dense_eval_score_random_baseline"] = float(random_mean)
    df["dense_eval_score_expert_baseline"] = float(expert_mean)
    df["dense_eval_score_rel_mean"] = 100.0 * (df["dense_eval_score_mean"] - random_mean) / denom
    df["dense_eval_score_rel_std"] = 100.0 * df["dense_eval_score_std"].fillna(0.0) / abs(denom)
    return df


def plot_dense_results(df, save_path):
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    x = df["epoch"].to_numpy()

    dense = df["dense_eval_score_rel_mean"]
    dense_smooth = smooth(dense)
    dense_std = df["dense_eval_score_rel_std"].fillna(0.0)
    axes[0].fill_between(
        x,
        (dense_smooth - dense_std).to_numpy(),
        (dense_smooth + dense_std).to_numpy(),
        color="#1f77b4",
        alpha=SHADE_ALPHA,
        linewidth=0,
    )
    axes[0].plot(x, dense_smooth, color="#1f77b4", linewidth=2.3, label="Random-Normalized Dense Score")
    axes[0].axhline(0.0, color="#666666", linestyle="--", linewidth=1.2, label="Random baseline")
    axes[0].axhline(100.0, color="#111111", linestyle=":", linewidth=1.2, label="Expert baseline")
    axes[0].set_title("Policy Return Under Dense Evaluation Reward", fontsize=15, fontweight="bold")
    axes[0].set_ylabel("Relative Score (random=0, expert=100)", fontsize=12)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(loc="best", fontsize=10)

    axes[1].plot(x, df["merge_success_rate"] * 100.0, color="#2ca02c", linewidth=2.0, label="Merge Success (%)")
    axes[1].plot(x, df["endpoint_success_rate"] * 100.0, color="#ff7f0e", linewidth=2.0, label="Endpoint Success (%)")
    axes[1].plot(x, df["safety_success_rate"] * 100.0, color="#d62728", linewidth=2.0, label="Safety Success (%)")
    axes[1].set_title("Success Metrics", fontsize=15, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Rate (%)", fontsize=12)
    axes[1].set_ylim(0, 105)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(loc="best", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved dense evaluation plot to: {save_path}")
    plt.show()


def main():
    cfg = Config()
    data_paths = [
        os.path.join(REPO_ROOT, "data", "lane_change_trajectories-0750am-0805am"),
        os.path.join(REPO_ROOT, "data", "lane_change_trajectories-0805am-0820am"),
        os.path.join(REPO_ROOT, "data", "lane_change_trajectories-0820am-0835am"),
    ]
    dataset = MergingDataset(data_paths, device="cpu")
    _, val_dataset = split_dataset(dataset, train_ratio=0.8, seed=cfg.SEED)

    print("[*] Evaluating random baseline...")
    random_policy = RandomPolicy(seed=cfg.SEED)
    random_stats = evaluate_dense_score(
        random_policy,
        val_dataset,
        cfg,
        n_eval_episodes=N_EVAL_EPISODES,
    )
    print("[*] Evaluating expert baseline...")
    expert_stats = evaluate_dense_score(
        None,
        val_dataset,
        cfg,
        n_eval_episodes=N_EVAL_EPISODES,
        action_provider=expert_action_provider,
    )
    random_mean = random_stats["dense_eval_score_mean"]
    expert_mean = expert_stats["dense_eval_score_mean"]
    print(
        "[*] Dense score baselines | "
        f"random={random_mean:.3f}, expert={expert_mean:.3f}, "
        f"denom={expert_mean - random_mean:.3f}"
    )

    checkpoints = []
    for path in glob.glob(CHECKPOINT_GLOB):
        epoch = checkpoint_epoch(path)
        if epoch is not None:
            checkpoints.append((epoch, path))
    checkpoints = sorted(checkpoints)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found: {CHECKPOINT_GLOB}")

    rows = []
    rows.append({"epoch": 0, "source": "random_baseline", **random_stats})
    for epoch, path in checkpoints:
        print(f"[*] Evaluating epoch {epoch}: {path}")
        model = PPO.load(path, device="cpu")
        row = {"epoch": epoch, "source": "checkpoint"}
        row.update(evaluate_dense_score(model, val_dataset, cfg, n_eval_episodes=N_EVAL_EPISODES))
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("epoch")
    df = add_random_normalized_score(df, random_mean=random_mean, expert_mean=expert_mean)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved dense evaluation table to: {OUTPUT_CSV}")
    plot_dense_results(df, OUTPUT_FIG)


if __name__ == "__main__":
    main()
