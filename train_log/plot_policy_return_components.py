import os

import matplotlib.pyplot as plt
import pandas as pd


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False

# Current environment reward budget:
# merge +0.5, endpoint +0.5, safety +1.0, total upper bound = 2.0.
MERGE_REWARD = 0.5
ENDPOINT_REWARD = 0.5
SAFETY_REWARD = 1.0
REWARD_BUDGET = MERGE_REWARD + ENDPOINT_REWARD + SAFETY_REWARD
NORM_SCALE = 100.0 / REWARD_BUDGET

SMOOTH_WINDOW = 3
SHADE_ALPHA = 0.16


def get_series(df, name):
    if name not in df.columns:
        raise KeyError(f"Missing required column: {name}")
    return pd.to_numeric(df[name], errors="coerce")


def smooth(series, window=SMOOTH_WINDOW):
    return series.rolling(window=window, min_periods=1, center=True).mean()


def rolling_bounds(series, window=SMOOTH_WINDOW):
    mean = smooth(series, window=window)
    std = series.rolling(window=window, min_periods=2, center=True).std().fillna(0.0)
    return mean, mean - std, mean + std


def plot_line_with_shadow(ax, x, series, *, color, label, linewidth=2.2):
    mean, lower, upper = rolling_bounds(series)
    ax.fill_between(x, lower.to_numpy(), upper.to_numpy(), color=color, alpha=SHADE_ALPHA, linewidth=0)
    ax.plot(x, mean, color=color, linewidth=linewidth, label=label)


def build_policy_return_components(df):
    epoch = get_series(df, "epoch")
    merge_rate = get_series(df, "merge_success_rate")
    endpoint_rate = get_series(df, "endpoint_success_rate")
    safety_rate = get_series(df, "safety_success_rate")
    eval_return_raw = get_series(df, "eval_ep_rew_mean")

    merge_raw = MERGE_REWARD * merge_rate
    endpoint_raw = ENDPOINT_REWARD * endpoint_rate
    safety_raw = SAFETY_REWARD * safety_rate
    component_sum_raw = merge_raw + endpoint_raw + safety_raw

    out = pd.DataFrame(
        {
            "epoch": epoch,
            "policy_return_raw": eval_return_raw,
            "policy_return_norm100": eval_return_raw * NORM_SCALE,
            "merge_component_raw": merge_raw,
            "endpoint_component_raw": endpoint_raw,
            "safety_component_raw": safety_raw,
            "component_sum_raw": component_sum_raw,
            "merge_component_norm100": merge_raw * NORM_SCALE,
            "endpoint_component_norm100": endpoint_raw * NORM_SCALE,
            "safety_component_norm100": safety_raw * NORM_SCALE,
            "component_sum_norm100": component_sum_raw * NORM_SCALE,
        }
    )
    return out


def plot_policy_return_components(eval_csv_path, save_dir, smooth_window=SMOOTH_WINDOW):
    global SMOOTH_WINDOW
    SMOOTH_WINDOW = int(smooth_window)

    if not os.path.exists(eval_csv_path):
        raise FileNotFoundError(f"eval_metrics.csv not found: {eval_csv_path}")

    df = pd.read_csv(eval_csv_path)
    comp = build_policy_return_components(df)
    x = comp["epoch"].to_numpy()

    max_abs_diff = (comp["policy_return_raw"] - comp["component_sum_raw"]).abs().max()
    print(f"[*] Environment reward budget = {REWARD_BUDGET:.1f}")
    print(f"[*] Normalized policy return: R_norm100 = {NORM_SCALE:.1f} * eval_ep_rew_mean")
    print(f"[*] Max |eval_ep_rew_mean - component_sum| = {max_abs_diff:.6f}")

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Panel 1: total policy return, scaled to 0-100.
    plot_line_with_shadow(
        axes[0],
        x,
        comp["policy_return_norm100"],
        color="#1f77b4",
        label="Policy Return (env reward, normalized to 0-100)",
    )
    axes[0].plot(
        x,
        comp["component_sum_norm100"],
        color="#111111",
        linewidth=1.6,
        linestyle="--",
        label="Merge + Endpoint + Safety components",
    )
    axes[0].set_title("Policy Return Under Environment Reward", fontsize=15, fontweight="bold")
    axes[0].set_ylabel("Normalized Return (0-100)", fontsize=12)
    axes[0].set_ylim(0, 105)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(loc="lower right", fontsize=10)

    # Panel 2: separated normalized component contributions.
    plot_line_with_shadow(
        axes[1],
        x,
        comp["merge_component_norm100"],
        color="#2ca02c",
        label="Merge contribution: 25 * merge_success_rate",
    )
    plot_line_with_shadow(
        axes[1],
        x,
        comp["endpoint_component_norm100"],
        color="#ff7f0e",
        label="Endpoint contribution: 25 * endpoint_success_rate",
    )
    plot_line_with_shadow(
        axes[1],
        x,
        comp["safety_component_norm100"],
        color="#d62728",
        label="Safety contribution: 50 * safety_success_rate",
    )
    axes[1].set_title("Policy Return Components", fontsize=15, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Normalized Return Points", fontsize=12)
    axes[1].set_ylim(0, 55)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(loc="best", fontsize=10)

    plt.tight_layout()

    csv_save_path = os.path.join(save_dir, "Policy_Return_Components_Norm100.csv")
    fig_save_path = os.path.join(save_dir, "Policy_Return_Components_Norm100.png")
    comp.to_csv(csv_save_path, index=False)
    plt.savefig(fig_save_path, dpi=300, bbox_inches="tight")
    print(f"Saved component table to: {csv_save_path}")
    print(f"Saved policy return component plot to: {fig_save_path}")
    plt.show()


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    eval_csv = os.path.join(
        script_dir,
        "baseline_attn_goal_safe_branch_aux_20260416_220623",
        "eval_metrics.csv",
    )
    plot_policy_return_components(eval_csv, script_dir)
