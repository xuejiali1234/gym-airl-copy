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
from evaluation.performance_evaluate_transfer_shield import build_shield_params
from evaluation.safety_shield_evaluate import shield_action

try:
    matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
    matplotlib.rcParams["axes.unicode_minus"] = False
except Exception:
    pass


def find_latest_bare_summary():
    base_dir = os.path.join(root_dir, "data-I80", "轨迹绘制")
    if not os.path.isdir(base_dir):
        return "", {}

    candidates = []
    for name in os.listdir(base_dir):
        path = os.path.join(base_dir, name)
        if os.path.isdir(path) and name.startswith("U220_D230_epoch290_transfer_"):
            summary_path = os.path.join(path, "trajectory_summary.csv")
            if os.path.exists(summary_path):
                candidates.append((os.path.getmtime(path), summary_path))

    if not candidates:
        return "", {}

    _, summary_path = max(candidates, key=lambda item: item[0])
    df = pd.read_csv(summary_path)
    mapping = {}
    for _, row in df.iterrows():
        mapping[str(row["filename"])] = {
            "bare_plot_path": row.get("plot_path", ""),
            "bare_collision": row.get("collision", ""),
            "bare_endpoint_success": row.get("endpoint_success", ""),
        }
    return summary_path, mapping


def run_inference_with_shield(model, env, shield_params):
    model_x, model_y, model_v = [], [], []
    intervention_indices = []
    warning_indices = []

    obs, _ = env.reset()
    px, py, vx, vy = env.ego_state
    model_x.append(px)
    model_y.append(py)
    model_v.append(np.sqrt(vx**2 + vy**2))

    done = False
    truncated = False
    max_steps = len(env.current_traj["ego_pos"]) + 50
    step_count = 0
    shield_interventions = 0
    shield_warnings = 0
    consecutive_interventions = 0
    safe_steps = 0
    last_info = {}

    while not (done or truncated) and step_count < max_steps:
        policy_action, _ = model.predict(obs, deterministic=True)
        policy_action = np.asarray(policy_action, dtype=np.float32).reshape(-1)
        action, shield_info = shield_action(
            env=env,
            policy_action=policy_action,
            params=shield_params,
            step_count=step_count,
            shield_interventions=shield_interventions,
            consecutive_interventions=consecutive_interventions,
            safe_steps=safe_steps,
        )

        if shield_info["intervened"]:
            shield_interventions += 1
            consecutive_interventions += 1
        else:
            consecutive_interventions = 0

        if shield_info.get("warning", False):
            shield_warnings += 1

        obs, reward, done, truncated, info = env.step(action)
        last_info = dict(info)

        px, py, vx, vy = env.ego_state
        speed = float(np.hypot(vx, vy))
        model_x.append(px)
        model_y.append(py)
        model_v.append(speed)

        if shield_info["intervened"]:
            intervention_indices.append(len(model_x) - 1)
        elif shield_info.get("warning", False):
            warning_indices.append(len(model_x) - 1)

        min_ttc = float(info.get("eval_min_ttc", 20.0))
        min_thw = float(info.get("eval_min_thw", 10.0))
        collided_now = bool(info.get("is_collided", False))
        if not collided_now and min_ttc >= shield_params.get("lead_ttc_min", 3.0) and min_thw >= shield_params.get(
            "follow_thw_min", 0.8
        ):
            safe_steps += 1
        else:
            safe_steps = 0

        step_count += 1

    collided = bool(getattr(env, "has_collided_this_episode", False))
    endpoint_success = bool(last_info.get("is_endpoint_success", False))
    merge_success = bool(last_info.get("is_merge_success", False))

    return {
        "model_x": model_x,
        "model_y": model_y,
        "model_v": model_v,
        "intervention_indices": intervention_indices,
        "warning_indices": warning_indices,
        "collision": collided,
        "endpoint_success": endpoint_success,
        "merge_success": merge_success,
        "shield_interventions": shield_interventions,
        "shield_warnings": shield_warnings,
        "shield_intervention_rate": shield_interventions / max(step_count, 1),
        "shield_warning_rate": shield_warnings / max(step_count, 1),
        "steps": step_count,
    }


def plot_one_trajectory(traj, rollout, cfg, save_path):
    gt_x = traj["ego_pos"][:, 0]
    gt_y = traj["ego_pos"][:, 1]
    gt_vx = traj["ego_vel"][:, 0]
    gt_vy = traj["ego_vel"][:, 1]
    gt_v = np.sqrt(gt_vx**2 + gt_vy**2)

    model_x = np.asarray(rollout["model_x"], dtype=np.float32)
    model_y = np.asarray(rollout["model_y"], dtype=np.float32)
    model_v = np.asarray(rollout["model_v"], dtype=np.float32)

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
        label="Policy + v10b",
        zorder=3,
        vmin=vmin_val,
        vmax=vmax_val,
    )
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Velocity (ft/s)", rotation=270, labelpad=15)

    if rollout["intervention_indices"]:
        idx = np.asarray(rollout["intervention_indices"], dtype=np.int32)
        ax.scatter(
            model_x[idx],
            model_y[idx],
            c="orange",
            marker="x",
            s=36,
            linewidths=1.0,
            label="Shield Intervene",
            zorder=4,
        )
    if rollout["warning_indices"]:
        idx = np.asarray(rollout["warning_indices"], dtype=np.int32)
        ax.scatter(
            model_x[idx],
            model_y[idx],
            facecolors="none",
            edgecolors="dodgerblue",
            marker="o",
            s=28,
            linewidths=0.8,
            label="Shield Warning",
            zorder=4,
        )

    ax.scatter(gt_x[0], gt_y[0], c="green", s=120, label="Start", zorder=5, edgecolors="white")
    ax.scatter(gt_x[-1], gt_y[-1], c="red", marker="X", s=120, label="End (GT)", zorder=5, edgecolors="white")
    ax.scatter(model_x[-1], model_y[-1], c="purple", marker="*", s=200, label="End (Shield)", zorder=6, edgecolors="white")

    status_bits = [
        f"merge={int(rollout['merge_success'])}",
        f"endpoint={int(rollout['endpoint_success'])}",
        f"collision={int(rollout['collision'])}",
        f"shield={rollout['shield_intervention_rate']:.2f}",
    ]
    title_color = "crimson" if rollout["collision"] else "black"
    ax.set_title(
        f"I80 Transfer + v10b\n{traj.get('filename', 'unknown')} | {' '.join(status_bits)}",
        fontsize=12,
        fontweight="bold",
        color=title_color,
        pad=18,
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
    return os.path.join(root_dir, "data-I80", "轨迹绘制1", f"U220_D230_epoch290_v10b_transfer_{stamp}")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize I80 transfer trajectories with v10b shield.")
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
    parser.add_argument("--shield-variant", default="v10b_leadgap_policy_veto_recovery")
    parser.add_argument("--prediction-horizon", type=float, default=1.0)
    parser.add_argument("--lead-ttc-min", type=float, default=3.0)
    parser.add_argument("--follow-ttc-min", type=float, default=3.0)
    parser.add_argument("--lead-thw-min", type=float, default=1.0)
    parser.add_argument("--follow-thw-min", type=float, default=0.8)
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
    shield_params = build_shield_params(
        variant=args.shield_variant,
        prediction_horizon=args.prediction_horizon,
        lead_ttc_min=args.lead_ttc_min,
        follow_ttc_min=args.follow_ttc_min,
        lead_thw_min=args.lead_thw_min,
        follow_thw_min=args.follow_thw_min,
    )

    total = len(eval_dataset)
    indices = list(range(total))
    if args.num_samples is not None and args.num_samples < total:
        indices = random.sample(indices, args.num_samples)

    bare_summary_path, bare_mapping = find_latest_bare_summary()

    os.makedirs(args.output_dir, exist_ok=True)
    rows = []

    print("=" * 80)
    print("I80 + v10b 迁移轨迹可视化开始")
    print(f"模型: {args.model_path}")
    print(f"评估集: {args.eval_data_path}")
    print(f"输出目录: {args.output_dir}")
    print(f"裸模型对照表: {bare_summary_path if bare_summary_path else '未找到'}")
    print(f"样本数: {len(indices)} / {total}")
    print("=" * 80)

    for idx in tqdm(indices, desc="Plot I80 + v10b"):
        traj = eval_dataset[idx]
        single_dataset = SingleTrajDataset(traj, expert_mean, expert_std)
        env = MergingEnv(single_dataset)
        env.collision_margin = 1.0

        rollout = run_inference_with_shield(model, env, shield_params)
        filename = traj.get("filename", f"trajectory_{idx}.csv")
        png_name = filename.replace(".csv", ".png")
        save_path = os.path.join(args.output_dir, png_name)
        plot_one_trajectory(traj, rollout, cfg, save_path)

        bare_info = bare_mapping.get(filename, {})
        rows.append(
            {
                "traj_index": idx,
                "filename": filename,
                "merge_success": int(rollout["merge_success"]),
                "endpoint_success": int(rollout["endpoint_success"]),
                "collision": int(rollout["collision"]),
                "shield_interventions": rollout["shield_interventions"],
                "shield_intervention_rate": rollout["shield_intervention_rate"],
                "shield_warnings": rollout["shield_warnings"],
                "shield_warning_rate": rollout["shield_warning_rate"],
                "steps": rollout["steps"],
                "v10b_plot_path": save_path,
                "bare_plot_path": bare_info.get("bare_plot_path", ""),
                "bare_collision": bare_info.get("bare_collision", ""),
                "bare_endpoint_success": bare_info.get("bare_endpoint_success", ""),
            }
        )

    df = pd.DataFrame(rows).sort_values(
        ["collision", "endpoint_success", "filename"],
        ascending=[False, True, True],
    )
    summary_csv = os.path.join(args.output_dir, "trajectory_summary.csv")
    comparison_csv = os.path.join(args.output_dir, "comparison_summary.csv")
    df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    df.to_csv(comparison_csv, index=False, encoding="utf-8-sig")

    print(f"Saved summary: {summary_csv}")
    print(f"Saved comparison: {comparison_csv}")
    print("I80 + v10b 迁移轨迹可视化完成。")


if __name__ == "__main__":
    main()
