#!/usr/bin/env python3
"""
Compare real US-101 lane-change trajectories against real I-80 lane-change
trajectories. This script does not include any model-generated distribution.

Usage
-----
python .\\evaluation\\distributions_comparison_i80.py
python .\\evaluation\\distributions_comparison_i80.py --no-save --no-show
"""

import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

try:
    import seaborn as sns
except ImportError:
    sns = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(curr_dir)
sys.path.insert(0, root_dir)

from configs.config import Config


cfg = Config()
FT_TO_M = 0.3048
CAR_LEN_FT = 15.0
N_BINS = 32


def configure_plot_style():
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            plt.style.use("ggplot")

    if sns is not None:
        sns.set_theme(context="talk", style="whitegrid")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "axes.unicode_minus": False,
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
            "figure.titlesize": 18,
            "grid.alpha": 0.28,
            "grid.linewidth": 0.8,
            "axes.linewidth": 1.0,
        }
    )


def compute_metrics(df):
    py = df["KF_Local_Y"].to_numpy(dtype=float)
    vx = df["KF_Vel_X"].to_numpy(dtype=float)
    vy = df["KF_Vel_Y"].to_numpy(dtype=float)

    if "KF_Acc_Y" in df.columns:
        ay = df["KF_Acc_Y"].to_numpy(dtype=float)
    else:
        ay = np.diff(vy, prepend=vy[0]) / cfg.DT

    speed = np.sqrt(vx**2 + vy**2) * FT_TO_M
    acc = ay * FT_TO_M

    l6_y = df["L6_Leading_Local_Y"].to_numpy(dtype=float)
    l6_v = df["L6_Leading_Vel"].to_numpy(dtype=float)
    l5_y = df["L5_Leading_Local_Y"].to_numpy(dtype=float)
    l5_v = df["L5_Leading_Vel"].to_numpy(dtype=float)

    if "Time_Hdwy" in df.columns:
        raw_thw = df["Time_Hdwy"].to_numpy(dtype=float)
        thw = raw_thw[np.isfinite(raw_thw) & (raw_thw > 0) & (raw_thw < 10)]
    else:
        thw = []

    ttc = []
    if not isinstance(thw, np.ndarray):
        thw = list(thw)

    for i in range(len(py)):
        candidates = []
        for lead_y, lead_v in ((l6_y[i], l6_v[i]), (l5_y[i], l5_v[i])):
            if lead_y == 0:
                continue
            gap = lead_y - py[i] - CAR_LEN_FT
            if gap > 0:
                candidates.append((gap, lead_v))

        if not candidates:
            continue

        dist, lead_v = min(candidates, key=lambda item: item[0])
        rel_v = vy[i] - lead_v

        if rel_v > 0.1:
            value = dist / rel_v
            if 0 < value < 20:
                ttc.append(value)

        if "Time_Hdwy" not in df.columns and vy[i] > 1.0:
            value = dist / vy[i]
            if 0 < value < 10:
                thw.append(value)

    return speed, acc, np.asarray(ttc), np.asarray(thw)


def sanitize(values, lower=None, upper=None):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if lower is not None:
        arr = arr[arr >= lower]
    if upper is not None:
        arr = arr[arr <= upper]
    return arr


def load_dataset(file_paths, label):
    print(f"[*] Loading {label} data from {len(file_paths)} files ...")
    metrics = {"speed": [], "acc": [], "ttc": [], "thw": []}

    for file_path in tqdm(file_paths, desc=label):
        try:
            df = pd.read_csv(file_path)
            speed, acc, ttc, thw = compute_metrics(df)
            metrics["speed"].append(speed)
            metrics["acc"].append(acc)
            metrics["ttc"].append(ttc)
            metrics["thw"].append(thw)
        except Exception as exc:
            print(f"[WARN] Failed on {os.path.basename(file_path)}: {exc}")

    for key in metrics:
        if metrics[key]:
            metrics[key] = np.concatenate(metrics[key])
        else:
            metrics[key] = np.array([], dtype=float)

    print(
        f"    {label}: speed={len(metrics['speed']):,}, "
        f"acc={len(metrics['acc']):,}, ttc={len(metrics['ttc']):,}, "
        f"thw={len(metrics['thw']):,}"
    )
    return metrics


def get_us101_files():
    pattern = os.path.join(root_dir, "data", "lane_change_trajectories-*", "vehicle_*.csv")
    file_paths = sorted(glob.glob(pattern))
    if not file_paths:
        raise FileNotFoundError(f"No US-101 files found with pattern: {pattern}")
    return file_paths


def get_i80_files():
    pattern = os.path.join(root_dir, "data-I80", "lane_change_trajectories-*", "vehicle_*.csv")
    file_paths = sorted(glob.glob(pattern))
    if not file_paths:
        raise FileNotFoundError(f"No I-80 files found with pattern: {pattern}")
    return file_paths


def pooled_limits(arr_a, arr_b, lower_q, upper_q, floor=None, fixed=None):
    if fixed is not None:
        return fixed
    pooled = np.concatenate([arr_a, arr_b])
    left, right = np.quantile(pooled, [lower_q, upper_q])
    span = max(right - left, 1e-6)
    pad = span * 0.08
    left -= pad
    right += pad
    if floor is not None:
        left = max(floor, left)
    return left, right


def draw_kde(ax, arr, color, xlim, fill=False, alpha=0.18, linewidth=2.2):
    if len(arr) < 2:
        return

    if len(arr) < 80:
        bins = np.linspace(xlim[0], xlim[1], 12)
        if fill:
            ax.hist(
                arr,
                bins=bins,
                density=True,
                histtype="stepfilled",
                alpha=alpha,
                color=color,
                edgecolor=color,
                linewidth=linewidth,
            )
        ax.hist(
            arr,
            bins=bins,
            density=True,
            histtype="step",
            color=color,
            linewidth=linewidth,
        )
        return

    if sns is not None:
        sns.kdeplot(
            arr,
            ax=ax,
            color=color,
            linewidth=linewidth,
            fill=fill,
            alpha=alpha if fill else 1.0,
            bw_adjust=1.05 if not fill else 1.1,
            cut=0,
            clip=xlim,
        )
        return

    xs = np.linspace(xlim[0], xlim[1], 400)
    kde = gaussian_kde(arr)
    ys = kde(xs)
    ax.plot(xs, ys, color=color, linewidth=linewidth)
    if fill:
        ax.fill_between(xs, 0, ys, color=color, alpha=alpha)


def plot_hist_kde(ax, arr_a, arr_b, xlabel, title, xlim, colors):
    bins = np.linspace(xlim[0], xlim[1], N_BINS + 1)
    for arr, color in zip([arr_a, arr_b], colors):
        ax.hist(
            arr,
            bins=bins,
            density=True,
            histtype="stepfilled",
            alpha=0.28,
            color=color,
            edgecolor=color,
            linewidth=1.1,
        )
        draw_kde(ax, arr, color, xlim, fill=False, linewidth=2.2)

    ax.set_xlim(*xlim)
    ax.set_title(title, pad=10, weight="semibold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")


def plot_kde(ax, arr_a, arr_b, xlabel, title, xlim, colors):
    for arr, color in zip([arr_a, arr_b], colors):
        draw_kde(ax, arr, color, xlim, fill=True, alpha=0.18, linewidth=2.4)

    ax.set_xlim(*xlim)
    ax.set_title(title, pad=10, weight="semibold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")


def build_distribution_figure(us101_data, i80_data):
    us_speed = sanitize(us101_data["speed"], lower=0.0)
    i80_speed = sanitize(i80_data["speed"], lower=0.0)
    us_acc = sanitize(us101_data["acc"], lower=-6.0, upper=6.0)
    i80_acc = sanitize(i80_data["acc"], lower=-6.0, upper=6.0)
    us_ttc = sanitize(us101_data["ttc"], lower=0.0, upper=15.0)
    i80_ttc = sanitize(i80_data["ttc"], lower=0.0, upper=15.0)
    us_thw = sanitize(us101_data["thw"], lower=0.0, upper=5.0)
    i80_thw = sanitize(i80_data["thw"], lower=0.0, upper=5.0)

    speed_xlim = pooled_limits(us_speed, i80_speed, 0.005, 0.995, floor=0.0)
    acc_xlim = pooled_limits(us_acc, i80_acc, 0.005, 0.995, fixed=(-4.0, 4.0))
    ttc_xlim = pooled_limits(us_ttc, i80_ttc, 0.005, 0.995, fixed=(0.0, 15.0))
    thw_xlim = pooled_limits(us_thw, i80_thw, 0.005, 0.995, fixed=(0.0, 5.0))

    colors = ["#356c9b", "#cc7a29"]
    labels = ["US-101", "I-80"]

    fig, axes = plt.subplots(2, 2, figsize=(16.5, 12.0), constrained_layout=True)

    plot_hist_kde(
        axes[0, 0], us_speed, i80_speed, "Speed (m/s)", "(a) Speed Distribution", speed_xlim, colors
    )
    plot_hist_kde(
        axes[0, 1], us_acc, i80_acc, "Acceleration (m/s^2)", "(b) Acceleration Distribution", acc_xlim, colors
    )
    plot_kde(
        axes[1, 0], us_ttc, i80_ttc, "Time to Collision (s)", "(c) TTC Distribution", ttc_xlim, colors
    )
    plot_kde(
        axes[1, 1], us_thw, i80_thw, "Time Headway (s)", "(d) Time Headway Distribution", thw_xlim, colors
    )

    handles = [
        plt.Line2D([0], [0], color=colors[0], linewidth=2.6),
        plt.Line2D([0], [0], color=colors[1], linewidth=2.6),
    ]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=True,
        bbox_to_anchor=(0.5, 1.01),
    )

    for ax in axes.ravel():
        ax.grid(True, alpha=0.28)

    return fig, {
        "US-101": {"speed": us_speed, "acc": us_acc, "ttc": us_ttc, "thw": us_thw},
        "I-80": {"speed": i80_speed, "acc": i80_acc, "ttc": i80_ttc, "thw": i80_thw},
    }


def save_distribution_table(data_dict, csv_path):
    rows = []
    for source_name, metrics in data_dict.items():
        for metric_name, values in metrics.items():
            for value in values:
                rows.append(
                    {"source": source_name, "metric": metric_name, "value": float(value)}
                )
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare US-101 and I-80 distributions.")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(curr_dir, "distribution_comparison_results_i80"),
        help="Directory to save figure and CSV.",
    )
    parser.add_argument("--no-save", action="store_true", help="Build figure but do not save.")
    parser.add_argument("--no-show", action="store_true", help="Build figure but do not show.")
    return parser.parse_args()


def main():
    args = parse_args()
    configure_plot_style()

    us101_data = load_dataset(get_us101_files(), "US-101")
    i80_data = load_dataset(get_i80_files(), "I-80")

    fig, export_data = build_distribution_figure(us101_data, i80_data)

    if not args.no_save:
        os.makedirs(args.output_dir, exist_ok=True)
        png_path = os.path.join(args.output_dir, "US101_vs_I80_Distribution.png")
        csv_path = os.path.join(args.output_dir, "US101_vs_I80_distribution_data.csv")
        fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
        save_distribution_table(export_data, csv_path)
        print(f"[OK] Figure saved to: {png_path}")
        print(f"[OK] Distribution data saved to: {csv_path}")

    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
