import os

import matplotlib.pyplot as plt
import pandas as pd


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False

SMOOTH_WINDOW = 9
SHADE_ALPHA = 0.18


def get_series(df, names):
    for name in names:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
    return None


def smooth_series(series, window=SMOOTH_WINDOW):
    return series.rolling(window=window, min_periods=1, center=True).mean()


def rolling_bounds(series, window=SMOOTH_WINDOW):
    mean = smooth_series(series, window=window)
    std = series.rolling(window=window, min_periods=2, center=True).std().fillna(0.0)
    return mean, mean - std, mean + std


def plot_with_shadow(ax, x, series, *, color, label):
    mean, lower, upper = rolling_bounds(series)
    ax.fill_between(x, lower.to_numpy(), upper.to_numpy(), color=color, alpha=SHADE_ALPHA, linewidth=0)
    ax.plot(x, mean, color=color, linewidth=2.0, label=label)


def plot_reward_means(csv_path, save_dir, smooth_window=SMOOTH_WINDOW):
    global SMOOTH_WINDOW
    SMOOTH_WINDOW = int(smooth_window)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"progress.csv not found: {csv_path}")

    df = pd.read_csv(csv_path)
    expert_reward = get_series(df, ["mean/disc/disc_reward_expert_mean", "disc_reward_expert_mean"])
    gen_reward = get_series(df, ["mean/disc/disc_reward_gen_mean", "disc_reward_gen_mean"])

    if expert_reward is None or gen_reward is None:
        raise ValueError(
            "This progress.csv does not contain disc_reward_expert_mean / "
            "disc_reward_gen_mean. Please use a log produced after adding reward gap logging."
        )

    x = list(range(1, len(df) + 1))
    fig, ax = plt.subplots(figsize=(12, 7))

    plot_with_shadow(ax, x, expert_reward, color="#1f77b4", label="Expert Reward Mean")
    plot_with_shadow(ax, x, gen_reward, color="#d62728", label="Generator Reward Mean")
    ax.axhline(y=0.0, color="gray", linestyle="-.", linewidth=1.2)

    ax.set_title(f"Discriminator Raw Reward Means (Rolling Window = {SMOOTH_WINDOW})", fontsize=15, fontweight="bold")
    ax.set_xlabel("Iterations", fontsize=12)
    ax.set_ylabel("Raw Reward Mean", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="best", fontsize=11)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "Disc_Reward_Expert_vs_Generator_Mean.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved reward mean comparison to: {save_path}")
    plt.show()


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_absolute_path = os.path.join(
        script_dir,
        "baseline_attn_goal_safe_branch_aux_20260417_224439",
        "progress.csv",
    )
    plot_reward_means(csv_absolute_path, script_dir)
