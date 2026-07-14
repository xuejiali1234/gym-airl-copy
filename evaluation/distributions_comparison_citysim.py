#!/usr/bin/env python3
"""
Compare real NGSIM (US-101) lane-change trajectories against aligned CitySim
lane-change trajectories. This script does not include model-generated data.

Notes
-----
- NGSIM data are loaded from ./data/lane_change_trajectories-*/
- CitySim data are loaded from ./data-CitySim-aligned/ when available.
- If the aligned export does not exist, the script falls back to raw
  ./data-CitySim/lane_change_trajectories-FreewayC-*/ directories and skips
  "_normalized" folders.
- The plotting style is tuned for paper figures: cleaner legends, consistent
  axis ranges, lighter fills, and less crowded typography.
"""

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
    px = df["KF_Local_X"].to_numpy(dtype=float)
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

    ttc_list = []
    if "Time_Hdwy" in df.columns:
        raw_thw = df["Time_Hdwy"].to_numpy(dtype=float)
        thw_list = raw_thw[np.isfinite(raw_thw) & (raw_thw > 0) & (raw_thw < 10)].tolist()
    else:
        thw_list = []

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

        if rel_v > 0.1 and dist > 0:
            ttc = dist / rel_v
            if 0 < ttc < 20:
                ttc_list.append(ttc)

        if "Time_Hdwy" not in df.columns and vy[i] > 1.0 and dist > 0:
            thw = dist / vy[i]
            if 0 < thw < 10:
                thw_list.append(thw)

    return speed, acc, np.asarray(ttc_list), np.asarray(thw_list)


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


def get_ngsim_files():
    pattern = os.path.join(root_dir, "data", "lane_change_trajectories-*", "vehicle_*.csv")
    file_paths = sorted(glob.glob(pattern))
    if not file_paths:
        raise FileNotFoundError(f"No NGSIM files found with pattern: {pattern}")
    return file_paths


def get_citysim_files():
    aligned_root = os.path.join(root_dir, "data-CitySim-aligned")
    raw_root = os.path.join(root_dir, "data-CitySim")

    if os.path.isdir(aligned_root):
        city_dirs = sorted(
            glob.glob(
                os.path.join(aligned_root, "lane_change_trajectories-FreewayC-*")
            )
        )
        city_dirs = [d for d in city_dirs if os.path.isdir(d)]
        source_tag = "aligned"
    else:
        city_dirs = sorted(
            glob.glob(os.path.join(raw_root, "lane_change_trajectories-FreewayC-*"))
        )
        city_dirs = [d for d in city_dirs if os.path.isdir(d) and not d.endswith("_normalized")]
        source_tag = "raw"

    file_paths = []
    for directory in city_dirs:
        file_paths.extend(glob.glob(os.path.join(directory, "vehicle_*.csv")))
    file_paths = sorted(file_paths)

    if not file_paths:
        raise FileNotFoundError("No CitySim trajectory files found.")
    return file_paths, source_tag


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


def build_distribution_figure(ng_data, cs_data):
    ng_speed = sanitize(ng_data["speed"], lower=0.0)
    cs_speed = sanitize(cs_data["speed"], lower=0.0)
    ng_acc = sanitize(ng_data["acc"], lower=-6.0, upper=6.0)
    cs_acc = sanitize(cs_data["acc"], lower=-6.0, upper=6.0)
    ng_ttc = sanitize(ng_data["ttc"], lower=0.0, upper=15.0)
    cs_ttc = sanitize(cs_data["ttc"], lower=0.0, upper=15.0)
    ng_thw = sanitize(ng_data["thw"], lower=0.0, upper=5.0)
    cs_thw = sanitize(cs_data["thw"], lower=0.0, upper=5.0)

    _, speed_right = pooled_limits(ng_speed, cs_speed, 0.005, 0.995, floor=0.0)
    speed_xlim = (0.0, speed_right)
    acc_xlim = pooled_limits(ng_acc, cs_acc, 0.005, 0.995, fixed=(-4.0, 4.0))
    ttc_xlim = pooled_limits(ng_ttc, cs_ttc, 0.005, 0.995, fixed=(0.0, 15.0))
    thw_xlim = pooled_limits(ng_thw, cs_thw, 0.005, 0.995, fixed=(0.0, 5.0))

    colors = ["#356c9b", "#cf5b5b"]
    labels = ["NGSIM", "CitySim"]

    fig, axes = plt.subplots(2, 2, figsize=(16.5, 12.0), constrained_layout=True)

    plot_hist_kde(
        axes[0, 0],
        ng_speed,
        cs_speed,
        "Speed (m/s)",
        "(a) Speed Distribution",
        speed_xlim,
        colors,
    )
    plot_hist_kde(
        axes[0, 1],
        ng_acc,
        cs_acc,
        "Acceleration (m/s^2)",
        "(b) Acceleration Distribution",
        acc_xlim,
        colors,
    )
    plot_kde(
        axes[1, 0],
        ng_ttc,
        cs_ttc,
        "Time to Collision (s)",
        "(c) TTC Distribution",
        ttc_xlim,
        colors,
    )
    plot_kde(
        axes[1, 1],
        ng_thw,
        cs_thw,
        "Time Headway (s)",
        "(d) Time Headway Distribution",
        thw_xlim,
        colors,
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
        "NGSIM": {"speed": ng_speed, "acc": ng_acc, "ttc": ng_ttc, "thw": ng_thw},
        "CitySim": {"speed": cs_speed, "acc": cs_acc, "ttc": cs_ttc, "thw": cs_thw},
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


def main():
    configure_plot_style()

    ngsim_files = get_ngsim_files()
    citysim_files, source_tag = get_citysim_files()

    ng_data = load_dataset(ngsim_files, "NGSIM")
    cs_label = "CitySim aligned" if source_tag == "aligned" else "CitySim raw"
    cs_data = load_dataset(citysim_files, cs_label)

    fig, export_data = build_distribution_figure(ng_data, cs_data)

    out_dir = os.path.join(curr_dir, "distribution_comparison_results_citysim")
    os.makedirs(out_dir, exist_ok=True)

    if source_tag == "aligned":
        png_path = os.path.join(out_dir, "NGSIM_vs_CitySim_Aligned_Distribution.png")
        csv_path = os.path.join(
            out_dir, "NGSIM_vs_CitySim_aligned_distribution_data.csv"
        )
    else:
        png_path = os.path.join(out_dir, "NGSIM_vs_CitySim_Distribution_Optimized.png")
        csv_path = os.path.join(out_dir, "NGSIM_vs_CitySim_distribution_data_optimized.csv")

    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    save_distribution_table(export_data, csv_path)
    print(f"[OK] CitySim source: {source_tag}")
    print(f"[OK] Figure saved to: {png_path}")
    print(f"[OK] Distribution data saved to: {csv_path}")
    plt.show()


if __name__ == "__main__":
    main()
