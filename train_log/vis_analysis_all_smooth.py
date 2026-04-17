import os

import pandas as pd
import matplotlib.pyplot as plt


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False

SMOOTH_WINDOW = 9
SHADE_ALPHA = 0.18
REWARD_SHIFT_C = 2.6
REWARD_DISPLAY_MODE = "zero_flip"  # "shift" or "zero_flip"


def get_column(df, possible_names):
    """Return the first existing column name from a list of candidates."""
    for name in possible_names:
        if name in df.columns:
            return name
    return None


def get_series(df, possible_names):
    """Return a numeric series for the first matching column, or None."""
    col = get_column(df, possible_names)
    if col is None:
        return None
    return pd.to_numeric(df[col], errors="coerce")


def smooth_series(series, window=SMOOTH_WINDOW):
    """Apply centered rolling-window smoothing while keeping edge points."""
    if series is None:
        return None
    return series.rolling(window=window, min_periods=1, center=True).mean()


def rolling_shadow(series, window=SMOOTH_WINDOW):
    """Return lower/upper rolling mean +/- std bounds for the shaded window."""
    smoothed = smooth_series(series, window=window)
    std = series.rolling(window=window, min_periods=2, center=True).std().fillna(0.0)
    return smoothed - std, smoothed + std


def shifted_episode_reward(df, reward, shift_c=REWARD_SHIFT_C):
    """Apply visualization-only AIRL reward shift: R_ep' = R_ep + c * L_ep."""
    ep_len = get_series(df, ["mean/gen/rollout/ep_len_mean", "rollout/ep_len_mean"])
    if ep_len is None:
        print("[!] Episode length column not found; plotting unshifted wrapped reward.")
        return reward

    shifted = reward + float(shift_c) * ep_len
    per_step = (reward / ep_len).replace([float("inf"), float("-inf")], pd.NA)
    tail_median = per_step.dropna().tail(50).median()
    print(f"[*] Reward display shift: R_ep' = R_ep + {shift_c:.3f} * episode_len")
    print(f"[*] Tail median original per-step reward: {tail_median:.3f}")
    return shifted


def zero_flipped_episode_reward(reward):
    """Mirror wrapped episode reward across zero: R_display = -R_ep."""
    print("[*] Reward display mode: zero-anchored flip, R_display = -R_ep")
    return -reward


def display_episode_reward(df, reward):
    """Choose the visualization-only reward transform for Figure 1."""
    if REWARD_DISPLAY_MODE == "shift":
        return shifted_episode_reward(df, reward, shift_c=REWARD_SHIFT_C)
    if REWARD_DISPLAY_MODE == "zero_flip":
        return zero_flipped_episode_reward(reward)
    print(f"[!] Unknown reward display mode '{REWARD_DISPLAY_MODE}', plotting raw reward.")
    return reward


def plot_smoothed(ax, iterations, series, *, color, linewidth=2.0, label=None, linestyle="-"):
    """Plot one smoothed metric line with a rolling std shadow."""
    smoothed = smooth_series(series)
    lower, upper = rolling_shadow(series)
    ax.fill_between(
        list(iterations),
        lower.to_numpy(),
        upper.to_numpy(),
        color=color,
        alpha=SHADE_ALPHA,
        linewidth=0,
    )
    ax.plot(
        iterations,
        smoothed,
        color=color,
        linewidth=linewidth,
        label=label,
        linestyle=linestyle,
    )


def decorate_axis(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)


def plot_combined_figures(csv_path, save_dir, smooth_window=SMOOTH_WINDOW):
    """Read progress.csv and draw the same 3x3 dashboard with smoothed curves."""
    global SMOOTH_WINDOW
    SMOOTH_WINDOW = int(smooth_window)

    print(f"[*] Reading data file: {csv_path}")
    print(f"[*] Applying centered rolling smoothing window: {SMOOTH_WINDOW}")
    print(f"[*] Drawing rolling mean +/- std shadow bands")
    print(f"[*] Reward display mode: {REWARD_DISPLAY_MODE}")
    if REWARD_DISPLAY_MODE == "shift":
        print(f"[*] Visualization-only reward shift c: {REWARD_SHIFT_C}")

    if not os.path.exists(csv_path):
        print(f"[!] Data file not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    iterations = range(1, len(df) + 1)

    fig, axes = plt.subplots(3, 3, figsize=(20, 15))

    # Figure 1: Episode Reward
    reward = get_series(
        df,
        ["mean/gen/rollout/ep_rew_wrapped_mean", "rollout/ep_rew_wrapped_mean"],
    )
    if reward is not None:
        reward = display_episode_reward(df, reward)
        plot_smoothed(axes[0, 0], iterations, reward, color="#1f77b4")
        reward_title = (
            f"Figure 1: Shifted Episode Reward (c={REWARD_SHIFT_C})"
            if REWARD_DISPLAY_MODE == "shift"
            else "Figure 1: Zero-Flipped Episode Reward"
        )
        decorate_axis(
            axes[0, 0],
            reward_title,
            "Iterations",
            "Displayed Reward",
        )
    else:
        axes[0, 0].axis("off")

    # Figure 2: Discriminator Loss
    disc_loss = get_series(df, ["mean/disc/disc_loss"])
    if disc_loss is not None:
        plot_smoothed(axes[0, 1], iterations, disc_loss, color="#d62728")
        decorate_axis(axes[0, 1], "Figure 2: Discriminator Loss", "Iterations", "Loss")
    else:
        axes[0, 1].axis("off")

    # Figure 3: Generator Value Loss
    value_loss = get_series(df, ["mean/gen/train/value_loss"])
    if value_loss is not None:
        plot_smoothed(axes[0, 2], iterations, value_loss, color="#2ca02c")
        decorate_axis(axes[0, 2], "Figure 3: Generator Value Loss", "Iterations", "Value Loss")
    else:
        axes[0, 2].axis("off")

    # Figure 4: Policy Loss
    policy_loss = get_series(df, ["mean/gen/train/policy_gradient_loss", "mean/gen/train/loss"])
    if policy_loss is not None:
        plot_smoothed(axes[1, 0], iterations, policy_loss, color="#ff7f0e")
        decorate_axis(axes[1, 0], "Figure 4: Policy Loss", "Iterations", "Policy Loss")
    else:
        axes[1, 0].axis("off")

    # Figure 5: Adversarial Accuracy
    acc_exp = get_series(df, ["mean/disc/disc_acc_expert"])
    acc_gen = get_series(df, ["mean/disc/disc_acc_gen"])
    if acc_exp is not None and acc_gen is not None:
        plot_smoothed(axes[1, 1], iterations, acc_exp, color="#1f77b4", label="Disc (Exp)")
        plot_smoothed(
            axes[1, 1],
            iterations,
            acc_gen,
            color="#d62728",
            label="Gen (Gen)",
            linestyle="--",
        )
        axes[1, 1].axhline(y=0.5, color="gray", linestyle="-.", label="0.5 Line")
        decorate_axis(axes[1, 1], "Figure 5: Adversarial Accuracy", "Iterations", "Accuracy")
        axes[1, 1].set_ylim(0, 1.05)
        axes[1, 1].legend(loc="lower right", fontsize=10)
    else:
        axes[1, 1].axis("off")

    # Figure 6: Action Standard Deviation
    action_std = get_series(df, ["mean/gen/train/std"])
    if action_std is not None:
        plot_smoothed(axes[1, 2], iterations, action_std, color="#8c564b")
        decorate_axis(axes[1, 2], "Figure 6: Action Std Deviation", "Iterations", "Action Std")
    else:
        axes[1, 2].axis("off")

    # Figure 7: Approx KL Divergence
    approx_kl = get_series(df, ["mean/gen/train/approx_kl"])
    if approx_kl is not None:
        plot_smoothed(axes[2, 0], iterations, approx_kl, color="#9467bd")
        decorate_axis(axes[2, 0], "Figure 7: Approx KL Divergence", "Iterations", "KL Divergence")
    else:
        axes[2, 0].axis("off")

    # Figure 8: Safety Auxiliary Loss
    safety_aux = get_series(df, ["mean/disc/disc_safety_aux", "disc_safety_aux"])
    if safety_aux is not None:
        plot_smoothed(axes[2, 1], iterations, safety_aux, color="#17becf")
        decorate_axis(axes[2, 1], "Figure 8: Safety Auxiliary Loss", "Iterations", "Safety Aux Loss")
    else:
        axes[2, 1].axis("off")

    # Figure 9: Reward Gap, with fallback for logs produced before this metric existed.
    reward_gap = get_series(df, ["mean/disc/disc_reward_gap", "disc_reward_gap"])
    if reward_gap is not None:
        plot_smoothed(axes[2, 2], iterations, reward_gap, color="#bcbd22")
        axes[2, 2].axhline(y=0.0, color="gray", linestyle="-.", linewidth=1.2)
        decorate_axis(axes[2, 2], "Figure 9: Reward Gap (Expert - Gen)", "Iterations", "Reward Gap")
    else:
        safety_reg = get_series(df, ["mean/disc/disc_safety_reg_loss", "disc_safety_reg_loss"])
        if safety_reg is None:
            safety_weight = get_series(df, ["mean/disc/disc_safety_weight", "disc_safety_weight"])
            if safety_aux is not None and safety_weight is not None:
                safety_reg = safety_aux * safety_weight

        if safety_reg is not None:
            plot_smoothed(axes[2, 2], iterations, safety_reg, color="#bcbd22")
            decorate_axis(axes[2, 2], "Figure 9: Safety Reg Loss (fallback)", "Iterations", "Reg Loss")
        else:
            axes[2, 2].axis("off")

    fig.suptitle(
        f"Smoothed Training Analysis (Rolling Window = {SMOOTH_WINDOW}, Reward Mode = {REWARD_DISPLAY_MODE})",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout(pad=3.0, rect=(0, 0, 1, 0.97))

    suffix = "Shifted_c2p6" if REWARD_DISPLAY_MODE == "shift" else "ZeroFlipped"
    save_path = os.path.join(save_dir, f"Full_Training_Analysis_9Figs_Smoothed_{suffix}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved smoothed 9-figure analysis dashboard to: {save_path}")
    plt.show()


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_absolute_path = os.path.join(script_dir, "baseline_attn_goal_safe_branch_aux_20260416_220623", "progress.csv")
    plot_combined_figures(csv_absolute_path, script_dir)
