import os
import sys

import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False

SMOOTH_WINDOW = 5
RUN_NAME = "baseline_attn_goal_safe_branch_aux_20260417_224439"
RUN_DIR = os.path.join(SCRIPT_DIR, RUN_NAME)
DEFAULT_EVAL_CSV = os.path.join(RUN_DIR, "eval_metrics.csv")


def smooth(series, window=SMOOTH_WINDOW):
    return series.rolling(window=window, min_periods=1, center=True).mean()


def get_series(df, name):
    if name not in df.columns:
        raise KeyError(f"Missing required eval metric column: {name}")
    return pd.to_numeric(df[name], errors="coerce")


def plot_eval_dense_metrics(eval_csv_path, save_dir=None):
    if save_dir is None:
        save_dir = os.path.dirname(eval_csv_path)
    df = pd.read_csv(eval_csv_path)
    epoch = get_series(df, "epoch")

    dense_norm = get_series(df, "eval_dense_return_norm100")
    paper_rank = get_series(df, "paper_rank_score_mean")
    endpoint = get_series(df, "endpoint_success_rate") * 100.0
    safety = get_series(df, "safety_success_rate") * 100.0
    collision = get_series(df, "collision_rate") * 100.0

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(epoch, smooth(dense_norm), color="#1f77b4", linewidth=2.3, label="Dense Return (norm100)")
    axes[0].plot(epoch, smooth(paper_rank), color="#9467bd", linewidth=2.0, label="Paper Rank Score")
    axes[0].set_title("Eval-Only Dense Policy Return", fontsize=15, fontweight="bold")
    axes[0].set_ylabel("Score", fontsize=12)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(loc="best", fontsize=10)

    axes[1].plot(epoch, smooth(endpoint), color="#ff7f0e", linewidth=2.0, label="Endpoint Success (%)")
    axes[1].plot(epoch, smooth(safety), color="#2ca02c", linewidth=2.0, label="Safety Success (%)")
    axes[1].plot(epoch, smooth(collision), color="#d62728", linewidth=2.0, label="Collision Rate (%)")
    axes[1].set_title("Success and Collision Metrics", fontsize=15, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Rate (%)", fontsize=12)
    axes[1].set_ylim(0, 105)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(loc="best", fontsize=10)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "Eval_Dense_Policy_Return.png")
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
