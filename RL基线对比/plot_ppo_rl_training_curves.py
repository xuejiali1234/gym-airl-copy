import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
TRAIN_LOG_DIR = ROOT_DIR / "train_log"

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False


def parse_args():
    parser = argparse.ArgumentParser(description="Plot PPO-RL training curves from progress/eval logs.")
    parser.add_argument(
        "--run-dir",
        default="",
        help="Path to a PPO-RL run directory. Defaults to the latest ppo_rl_baseline_* directory.",
    )
    parser.add_argument("--window", type=int, default=11, help="Centered smoothing window.")
    parser.add_argument("--show", action="store_true", help="Display the figure after saving.")
    return parser.parse_args()


def latest_run_dir():
    candidates = sorted(
        [path for path in TRAIN_LOG_DIR.glob("ppo_rl_baseline_*") if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No ppo_rl_baseline_* directories were found under train_log/.")
    return candidates[0]


def find_run_dir(run_dir_arg):
    if run_dir_arg:
        run_dir = Path(run_dir_arg).expanduser().resolve()
    else:
        run_dir = latest_run_dir()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    progress_path = run_dir / "progress.csv"
    eval_path = run_dir / "eval_metrics.csv"
    if not progress_path.exists():
        raise FileNotFoundError(f"Missing progress.csv: {progress_path}")
    if not eval_path.exists():
        raise FileNotFoundError(f"Missing eval_metrics.csv: {eval_path}")
    return run_dir, progress_path, eval_path


def read_csv(path):
    df = pd.read_csv(path)
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass
    return df


def smooth(series, window):
    return series.rolling(window=window, min_periods=1, center=True).mean()


def band(series, window):
    center = smooth(series, window)
    std = series.rolling(window=window, min_periods=2, center=True).std().fillna(0.0)
    return center, center - std, center + std


def plot_with_band(ax, x, series, *, color, label=None, window=11, linewidth=2.2, alpha=0.16, linestyle="-"):
    clean = pd.to_numeric(series, errors="coerce")
    center, lower, upper = band(clean, window)
    ax.fill_between(x, lower.to_numpy(), upper.to_numpy(), color=color, alpha=alpha, linewidth=0)
    ax.plot(x, center.to_numpy(), color=color, linewidth=linewidth, label=label, linestyle=linestyle)


def style_axis(ax, title, xlabel="Epoch", ylabel=""):
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.tick_params(labelsize=10)


def draw_dashboard(run_dir, progress_df, eval_df, window):
    fig, axes = plt.subplots(3, 3, figsize=(19, 14))
    prog_x = progress_df["epoch"].to_numpy()
    eval_x = eval_df["epoch"].to_numpy()

    # 1. Train reward
    plot_with_band(
        axes[0, 0],
        prog_x,
        progress_df["train_reward_mean"],
        color="#1f77b4",
        window=window,
    )
    style_axis(axes[0, 0], "Figure 1: Train Episode Reward", ylabel="Reward")

    # 2. Eval success rates
    plot_with_band(axes[0, 1], eval_x, eval_df["merge_success_rate"], color="#2ca02c", label="Merge", window=window)
    plot_with_band(
        axes[0, 1], eval_x, eval_df["endpoint_success_rate"], color="#1f77b4", label="Endpoint", window=window
    )
    plot_with_band(
        axes[0, 1], eval_x, eval_df["safety_success_rate"], color="#9467bd", label="Safety", window=window
    )
    plot_with_band(axes[0, 1], eval_x, eval_df["collision_rate"], color="#d62728", label="Collision", window=window)
    style_axis(axes[0, 1], "Figure 2: Eval Success / Collision", ylabel="Rate")
    axes[0, 1].set_ylim(-0.02, 1.02)
    axes[0, 1].legend(loc="upper right", fontsize=9, frameon=True)

    # 3. Dense return / paper score
    plot_with_band(
        axes[0, 2], eval_x, eval_df["eval_dense_return_norm100"], color="#ff7f0e", label="Dense100", window=window
    )
    plot_with_band(
        axes[0, 2], eval_x, eval_df["paper_rank_score_mean"], color="#8c564b", label="Paper score", window=window
    )
    style_axis(axes[0, 2], "Figure 3: Eval Dense / Paper Score", ylabel="Score")
    axes[0, 2].legend(loc="best", fontsize=9, frameon=True)

    # 4. Reward components
    component_cols = [
        ("term_eff", "#2ca02c", "Eff"),
        ("term_safety", "#d62728", "Safety"),
        ("term_thw", "#9467bd", "THW"),
        ("term_comfort", "#7f7f7f", "Comfort"),
        ("term_goal", "#1f77b4", "Goal"),
    ]
    for col, color, label in component_cols:
        if col in progress_df.columns:
            plot_with_band(axes[1, 0], prog_x, progress_df[col], color=color, label=label, window=window)
    style_axis(axes[1, 0], "Figure 4: Train Reward Components", ylabel="Term value")
    axes[1, 0].legend(loc="best", fontsize=9, ncol=2, frameon=True)

    # 5. Event rates
    event_cols = [
        ("merge_bonus_rate", "#2ca02c", "Merge bonus"),
        ("success_bonus_rate", "#1f77b4", "Success bonus"),
        ("timeout_penalty_rate", "#ff7f0e", "Timeout"),
        ("collision_penalty_rate", "#d62728", "Collision"),
    ]
    for col, color, label in event_cols:
        plot_with_band(axes[1, 1], prog_x, progress_df[col], color=color, label=label, window=window)
    style_axis(axes[1, 1], "Figure 5: Terminal Event Rates", ylabel="Rate / step")
    axes[1, 1].legend(loc="best", fontsize=9, frameon=True)

    # 6. PPO optimization signals
    plot_with_band(
        axes[1, 2], prog_x, progress_df["train_value_loss"], color="#2ca02c", label="Value loss", window=window
    )
    plot_with_band(
        axes[1, 2],
        prog_x,
        progress_df["train_policy_gradient_loss"],
        color="#ff7f0e",
        label="Policy loss",
        window=window,
    )
    plot_with_band(
        axes[1, 2], prog_x, progress_df["train_approx_kl"], color="#9467bd", label="Approx KL", window=window
    )
    style_axis(axes[1, 2], "Figure 6: PPO Optimization Signals", ylabel="Value")
    axes[1, 2].legend(loc="best", fontsize=9, frameon=True)

    # 7. Train physical metrics
    plot_with_band(
        axes[2, 0], prog_x, progress_df["mean_speed_train_mps"], color="#1f77b4", label="Speed (m/s)", window=window
    )
    plot_with_band(
        axes[2, 0], prog_x, progress_df["mean_abs_jerk_train"], color="#d62728", label="|Jerk| (m/s^3)", window=window
    )
    style_axis(axes[2, 0], "Figure 7: Train Speed / Jerk", ylabel="Physical value")
    axes[2, 0].legend(loc="best", fontsize=9, frameon=True)

    # 8. Eval safety metrics
    plot_with_band(axes[2, 1], eval_x, eval_df["min_ttc"], color="#2ca02c", label="Min TTC", window=window)
    plot_with_band(axes[2, 1], eval_x, eval_df["min_thw"], color="#1f77b4", label="Min THW", window=window)
    plot_with_band(
        axes[2, 1], eval_x, eval_df["mean_abs_jerk"], color="#d62728", label="|Jerk|", window=window
    )
    style_axis(axes[2, 1], "Figure 8: Eval TTC / THW / Jerk", ylabel="Value")
    axes[2, 1].legend(loc="best", fontsize=9, frameon=True)

    # 9. Learning stability extras
    plot_with_band(
        axes[2, 2], prog_x, progress_df["rollout_ep_rew_mean"], color="#1f77b4", label="SB3 rollout reward", window=window
    )
    plot_with_band(
        axes[2, 2], prog_x, progress_df["rollout_ep_len_mean"], color="#8c564b", label="Rollout length", window=window
    )
    plot_with_band(
        axes[2, 2], prog_x, progress_df["train_clip_fraction"], color="#17becf", label="Clip fraction", window=window
    )
    style_axis(axes[2, 2], "Figure 9: Rollout / Clip Fraction", ylabel="Value")
    axes[2, 2].legend(loc="best", fontsize=9, frameon=True)

    fig.suptitle(
        f"PPO-RL Training Dashboard ({run_dir.name})",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout(rect=(0, 0, 1, 0.97), pad=2.2)
    return fig


def main():
    args = parse_args()
    run_dir, progress_path, eval_path = find_run_dir(args.run_dir)
    progress_df = read_csv(progress_path)
    eval_df = read_csv(eval_path)

    print(f"[*] Run dir: {run_dir}")
    print(f"[*] Progress csv: {progress_path}")
    print(f"[*] Eval csv: {eval_path}")
    print(f"[*] Smoothing window: {args.window}")

    fig = draw_dashboard(run_dir, progress_df, eval_df, args.window)
    save_path = run_dir / "PPO_RL_Training_Curves.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"[*] Saved figure to: {save_path}")
    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
