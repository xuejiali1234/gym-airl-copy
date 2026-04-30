import argparse
import os
import random
import sys
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.collections import LineCollection
from stable_baselines3 import PPO
from tqdm import tqdm

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from configs.config import Config
from envs.merging_env import MergingEnv
from evaluation.performance_evaluate_transfer import (
    SingleTrajDataset,
    build_default_model_path,
    build_default_stats_paths,
    load_eval_dataset_with_fixed_stats,
    load_stats_dataset,
)

try:
    matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
    matplotlib.rcParams["axes.unicode_minus"] = False
except Exception:
    pass


def run_inference(model, env):
    model_x, model_y, model_v = [], [], []
    obs, info = env.reset()

    px, py, vx, vy = env.ego_state
    model_x.append(px)
    model_y.append(py)
    model_v.append(np.sqrt(vx**2 + vy**2))

    done = False
    truncated = False
    max_steps = len(env.current_traj["ego_pos"]) + 50
    step_count = 0
    episode_collided = False
    is_endpoint_success = False

    while not (done or truncated) and step_count < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        episode_collided = episode_collided or bool(info.get("is_collided", False))

        px, py, vx, vy = env.ego_state
        model_x.append(px)
        model_y.append(py)
        model_v.append(np.sqrt(vx**2 + vy**2))
        step_count += 1

        if done or truncated:
            is_endpoint_success = info.get("is_endpoint_success", info.get("is_success", False))
            episode_collided = episode_collided or bool(getattr(env, "has_collided_this_episode", False))

    return {
        "model_x": model_x,
        "model_y": model_y,
        "model_v": model_v,
        "collision": episode_collided,
        "endpoint_success": bool(is_endpoint_success),
        "steps": step_count,
    }


def plot_one_trajectory(traj, rollout, cfg, save_path):
    gt_x = traj["ego_pos"][:, 0]
    gt_y = traj["ego_pos"][:, 1]
    gt_vx = traj["ego_vel"][:, 0]
    gt_vy = traj["ego_vel"][:, 1]
    gt_v = np.sqrt(gt_vx**2 + gt_vy**2)

    model_x = rollout["model_x"]
    model_y = rollout["model_y"]
    model_v = rollout["model_v"]

    fig, ax = plt.subplots(figsize=(7, 8.5))

    vmin_val, vmax_val = 0.0, 80.0
    y_min = min(np.min(gt_y), np.min(model_y)) - 20
    y_max = max(np.max(gt_y), np.max(model_y)) + 20

    boundary_style = {"color": "gray", "linestyle": "--", "linewidth": 1.0, "alpha": 0.7}
    ax.vlines([cfg.X_MIN, cfg.X_MAX], y_min, y_max, **boundary_style)
    ax.vlines(
        [cfg.X_MIN + cfg.LANE_WIDTH],
        y_min,
        y_max,
        color="black",
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
    )

    points = np.array([gt_x, gt_y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap="RdYlGn_r", linewidth=4.0, alpha=0.7, zorder=2)
    lc.set_array(gt_v)
    lc.set_clim(vmin=vmin_val, vmax=vmax_val)
    ax.add_collection(lc)
    ax.plot(gt_x, gt_y, color="black", linewidth=1.0, alpha=0.4, label="Ground Truth", zorder=2)

    sc = ax.scatter(
        model_x,
        model_y,
        c=model_v,
        cmap="RdYlGn_r",
        s=25,
        edgecolors="black",
        linewidths=0.5,
        label="Policy Rollout",
        zorder=3,
        vmin=vmin_val,
        vmax=vmax_val,
    )
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Velocity (ft/s)", rotation=270, labelpad=15)

    ax.scatter(gt_x[0], gt_y[0], c="green", s=120, label="Start", zorder=5, edgecolors="white")
    ax.scatter(gt_x[-1], gt_y[-1], c="red", marker="X", s=120, label="End (GT)", zorder=5, edgecolors="white")
    ax.scatter(model_x[-1], model_y[-1], c="purple", marker="*", s=200, label="End (Model)", zorder=6, edgecolors="white")

    status_bits = []
    status_bits.append(f"endpoint={int(rollout['endpoint_success'])}")
    status_bits.append(f"collision={int(rollout['collision'])}")

    title_color = "crimson" if rollout["collision"] else "black"
    ax.set_title(
        f"I80 Transfer Trajectory\n{traj.get('filename', 'unknown')} | {' '.join(status_bits)}",
        fontsize=12,
        fontweight="bold",
        color=title_color,
        pad=20,
    )
    ax.set_aspect(0.5)
    ax.set_xlim(cfg.X_MIN - 5, cfg.X_MAX + 5)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Lateral Position (ft)")
    ax.set_ylabel("Longitudinal Position (ft)")
    ax.legend(loc="upper right", bbox_to_anchor=(-0.08, 1.0), fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def default_output_dir():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(root_dir, "data-I80", "轨迹绘制", f"U220_D230_epoch290_transfer_{stamp}")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize I80 transfer trajectories.")
    parser.add_argument("--model-path", default=build_default_model_path())
    parser.add_argument("--stats-data-path", nargs="+", default=build_default_stats_paths())
    parser.add_argument(
        "--eval-data-path",
        nargs="+",
        default=[os.path.join(root_dir, "data-I80", "lane_change_trajectories-0400pm-0415pm")],
    )
    parser.add_argument("--output-dir", default=default_output_dir())
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--collision-margin", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, (expert_mean, expert_std) = load_stats_dataset(args.stats_data_path, device=device)
    eval_dataset = load_eval_dataset_with_fixed_stats(
        args.eval_data_path,
        expert_mean=expert_mean,
        expert_std=expert_std,
        device=device,
    )

    model = PPO.load(args.model_path, device=device)

    total = len(eval_dataset)
    indices = list(range(total))
    if args.num_samples is not None and args.num_samples < total:
        indices = random.sample(indices, args.num_samples)

    os.makedirs(args.output_dir, exist_ok=True)
    rows = []

    print("=" * 80)
    print("I80 迁移轨迹可视化开始")
    print(f"模型: {args.model_path}")
    print(f"评估集: {args.eval_data_path}")
    print(f"输出目录: {args.output_dir}")
    print(f"样本数: {len(indices)} / {total}")
    print("=" * 80)

    for idx in tqdm(indices, desc="Plot I80 transfer"):
        traj = eval_dataset[idx]
        single_dataset = SingleTrajDataset(traj, expert_mean, expert_std)
        env = MergingEnv(single_dataset)
        env.collision_margin = args.collision_margin

        rollout = run_inference(model, env)
        filename = traj.get("filename", f"trajectory_{idx}.csv")
        png_name = filename.replace(".csv", ".png")
        save_path = os.path.join(args.output_dir, png_name)
        plot_one_trajectory(traj, rollout, cfg, save_path)

        rows.append(
            {
                "traj_index": idx,
                "filename": filename,
                "collision": int(rollout["collision"]),
                "endpoint_success": int(rollout["endpoint_success"]),
                "steps": rollout["steps"],
                "plot_path": save_path,
            }
        )

    summary_csv = os.path.join(args.output_dir, "trajectory_summary.csv")
    pd.DataFrame(rows).sort_values(["collision", "endpoint_success", "filename"], ascending=[False, True, True]).to_csv(
        summary_csv,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Saved trajectory summary: {summary_csv}")
    print("I80 迁移轨迹可视化完成。")


if __name__ == "__main__":
    main()
