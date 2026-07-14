#!/usr/bin/env python3
"""
Diagnose and correct the CitySim longitudinal-direction mismatch for TTC/THW
analysis.

What this script does
---------------------
1. Checks whether CitySim raw CSV schema matches the original US-101 schema.
2. Diagnoses longitudinal-direction mismatch using raw trajectory statistics.
3. Recomputes TTC / THW with two CitySim modes:
   - raw signed longitudinal speed
   - aligned longitudinal speed magnitude
4. Plots US-101, CitySim raw, and CitySim aligned TTC/THW distributions.

Outputs
-------
- evaluation/citysim_alignment_diagnosis_results/citysim_alignment_summary.json
- evaluation/citysim_alignment_diagnosis_results/US101_vs_CitySim_TTC_THW_Aligned.png
"""

import argparse
import glob
import json
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

CAR_LEN_FT = 15.0


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
            "legend.fontsize": 12,
            "grid.alpha": 0.28,
            "grid.linewidth": 0.8,
            "axes.linewidth": 1.0,
        }
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose and align CitySim longitudinal direction for TTC/THW."
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(curr_dir, "citysim_alignment_diagnosis_results"),
        help="Directory for saved figure and summary JSON.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional file cap for quick diagnosis / smoke tests.",
    )
    parser.add_argument("--no-save", action="store_true", help="Do not save outputs.")
    parser.add_argument("--no-show", action="store_true", help="Do not display figure.")
    return parser.parse_args()


def get_us101_files(max_files=None):
    pattern = os.path.join(root_dir, "data", "lane_change_trajectories-*", "vehicle_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No US-101 files found with pattern: {pattern}")
    return files[:max_files] if max_files else files


def get_citysim_raw_files(max_files=None):
    files = []
    for directory in sorted(
        glob.glob(os.path.join(root_dir, "data-CitySim", "lane_change_trajectories-FreewayC-*"))
    ):
        if directory.endswith("_normalized"):
            continue
        files.extend(glob.glob(os.path.join(directory, "vehicle_*.csv")))
    files = sorted(files)
    if not files:
        raise FileNotFoundError("No CitySim raw trajectory files found.")
    return files[:max_files] if max_files else files


def compare_schema(us_file, city_file):
    us_df = pd.read_csv(us_file, nrows=5)
    city_df = pd.read_csv(city_file, nrows=5)
    us_cols = list(us_df.columns)
    city_cols = list(city_df.columns)
    return {
        "us101_col_count": len(us_cols),
        "citysim_raw_col_count": len(city_cols),
        "columns_match": us_cols == city_cols,
        "only_us101": [col for col in us_cols if col not in city_cols],
        "only_citysim": [col for col in city_cols if col not in us_cols],
        "us101_dtypes": {k: str(v) for k, v in us_df.dtypes.items()},
        "citysim_raw_dtypes": {k: str(v) for k, v in city_df.dtypes.items()},
    }


def safe_quantiles(values, q_list):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return [round(float(v), 4) for v in np.quantile(arr, q_list)]


def select_nearest_lead_gap(py, l6_y, l6_v, l5_y, l5_v):
    candidates = []
    for lead_y, lead_v in ((l6_y, l6_v), (l5_y, l5_v)):
        if lead_y == 0:
            continue
        gap = lead_y - py - CAR_LEN_FT
        if gap > 0:
            candidates.append((gap, lead_v))
    if not candidates:
        return None, None
    return min(candidates, key=lambda item: item[0])


def summarize_orientation(file_paths, dataset_name):
    summary = {
        "dataset": dataset_name,
        "files": len(file_paths),
        "rows": 0,
        "vy_positive": 0,
        "vy_negative": 0,
        "dy_positive": 0,
        "dy_negative": 0,
        "lead_nonzero": 0,
        "lead_gap_positive": 0,
        "lane_ids": set(),
    }
    vy_values = []

    for file_path in tqdm(file_paths, desc=f"{dataset_name} orientation"):
        df = pd.read_csv(file_path)
        py = pd.to_numeric(df["KF_Local_Y"], errors="coerce").to_numpy()
        vy = pd.to_numeric(df["KF_Vel_Y"], errors="coerce").to_numpy()
        l6_y = pd.to_numeric(df["L6_Leading_Local_Y"], errors="coerce").to_numpy()
        l5_y = pd.to_numeric(df["L5_Leading_Local_Y"], errors="coerce").to_numpy()
        lane = pd.to_numeric(df["Lane_ID"], errors="coerce").dropna().astype(int).tolist()

        summary["rows"] += len(df)
        summary["vy_positive"] += int(np.sum(vy > 1e-6))
        summary["vy_negative"] += int(np.sum(vy < -1e-6))
        dy = np.diff(py)
        summary["dy_positive"] += int(np.sum(dy > 1e-6))
        summary["dy_negative"] += int(np.sum(dy < -1e-6))
        summary["lane_ids"].update(lane)
        vy_values.append(vy[np.isfinite(vy)])

        for i in range(len(df)):
            lead_candidates = [l6_y[i], l5_y[i]]
            for lead_y in lead_candidates:
                if lead_y == 0:
                    continue
                summary["lead_nonzero"] += 1
                if (lead_y - py[i]) > 0:
                    summary["lead_gap_positive"] += 1

    vy_values = np.concatenate(vy_values) if vy_values else np.array([], dtype=float)
    rows = max(summary["rows"], 1)
    dy_total = max(summary["dy_positive"] + summary["dy_negative"], 1)
    lead_total = max(summary["lead_nonzero"], 1)

    return {
        "dataset": dataset_name,
        "files": summary["files"],
        "rows": summary["rows"],
        "lane_ids": sorted(summary["lane_ids"]),
        "vy_positive_rate": round(summary["vy_positive"] / rows, 4),
        "vy_negative_rate": round(summary["vy_negative"] / rows, 4),
        "y_increasing_rate": round(summary["dy_positive"] / dy_total, 4),
        "y_decreasing_rate": round(summary["dy_negative"] / dy_total, 4),
        "lead_ahead_by_positive_gap_rate": round(summary["lead_gap_positive"] / lead_total, 4),
        "median_vy": round(float(np.median(vy_values)), 4) if vy_values.size else None,
        "needs_longitudinal_speed_alignment": bool(
            vy_values.size
            and np.median(vy_values) < 0
            and (summary["lead_gap_positive"] / lead_total) > 0.5
        ),
    }


def collect_ttc_thw(file_paths, dataset_name, speed_mode):
    assert speed_mode in {"signed", "magnitude"}

    total_rows = 0
    candidate_rows = 0
    positive_rel_speed_rows = 0
    ttc_values = []
    thw_values = []
    rel_speed_values = []

    for file_path in tqdm(file_paths, desc=f"{dataset_name} {speed_mode}"):
        df = pd.read_csv(file_path)
        py = pd.to_numeric(df["KF_Local_Y"], errors="coerce").to_numpy()
        vy = pd.to_numeric(df["KF_Vel_Y"], errors="coerce").to_numpy()
        l6_y = pd.to_numeric(df["L6_Leading_Local_Y"], errors="coerce").to_numpy()
        l6_v = pd.to_numeric(df["L6_Leading_Vel"], errors="coerce").to_numpy()
        l5_y = pd.to_numeric(df["L5_Leading_Local_Y"], errors="coerce").to_numpy()
        l5_v = pd.to_numeric(df["L5_Leading_Vel"], errors="coerce").to_numpy()

        total_rows += len(df)

        for i in range(len(df)):
            gap, lead_v = select_nearest_lead_gap(py[i], l6_y[i], l6_v[i], l5_y[i], l5_v[i])
            if gap is None:
                continue

            candidate_rows += 1
            ego_long = vy[i] if speed_mode == "signed" else abs(vy[i])
            lead_long = lead_v if speed_mode == "signed" else abs(lead_v)
            rel_speed = ego_long - lead_long
            rel_speed_values.append(rel_speed)

            if rel_speed > 0.1:
                positive_rel_speed_rows += 1
                ttc = gap / rel_speed
                if 0 < ttc < 20:
                    ttc_values.append(ttc)

            if ego_long > 1.0:
                thw = gap / ego_long
                if 0 < thw < 10:
                    thw_values.append(thw)

    total_rows = max(total_rows, 1)
    candidate_rows_safe = max(candidate_rows, 1)

    return {
        "dataset": dataset_name,
        "speed_mode": speed_mode,
        "total_rows": total_rows,
        "candidate_rows": candidate_rows,
        "candidate_rate": round(candidate_rows / total_rows, 4),
        "positive_rel_speed_rate_all_rows": round(positive_rel_speed_rows / total_rows, 4),
        "positive_rel_speed_rate_candidate_rows": round(
            positive_rel_speed_rows / candidate_rows_safe, 4
        ),
        "ttc_count": len(ttc_values),
        "thw_count": len(thw_values),
        "ttc_rate_all_rows": round(len(ttc_values) / total_rows, 4),
        "thw_rate_all_rows": round(len(thw_values) / total_rows, 4),
        "rel_speed_q10_q50_q90": safe_quantiles(rel_speed_values, [0.1, 0.5, 0.9]),
        "ttc_q10_q50_q90": safe_quantiles(ttc_values, [0.1, 0.5, 0.9]),
        "thw_q10_q50_q90": safe_quantiles(thw_values, [0.1, 0.5, 0.9]),
        "ttc_values": np.asarray(ttc_values, dtype=float),
        "thw_values": np.asarray(thw_values, dtype=float),
    }


def sanitize(values, upper):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = arr[(arr > 0) & (arr <= upper)]
    return arr


def draw_density(ax, values, label, color, xlim, linestyle="-", fill=False, alpha=0.18):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return

    if arr.size < 80:
        bins = np.linspace(xlim[0], xlim[1], 12)
        ax.hist(
            arr,
            bins=bins,
            density=True,
            histtype="step",
            color=color,
            linewidth=2.0,
            linestyle=linestyle,
            label=label,
        )
        if fill:
            ax.hist(
                arr,
                bins=bins,
                density=True,
                histtype="stepfilled",
                color=color,
                alpha=alpha,
            )
        return

    if sns is not None:
        sns.kdeplot(
            arr,
            ax=ax,
            color=color,
            linewidth=2.2,
            linestyle=linestyle,
            fill=fill,
            alpha=alpha if fill else 1.0,
            bw_adjust=1.08,
            cut=0,
            clip=xlim,
            label=label,
        )
        return

    xs = np.linspace(xlim[0], xlim[1], 400)
    kde = gaussian_kde(arr)
    ys = kde(xs)
    ax.plot(xs, ys, color=color, linewidth=2.2, linestyle=linestyle, label=label)
    if fill:
        ax.fill_between(xs, 0, ys, color=color, alpha=alpha)


def build_figure(us101_metrics, city_raw_metrics, city_aligned_metrics):
    colors = {
        "us101": "#356c9b",
        "city_raw": "#c44e52",
        "city_aligned": "#3c9d5d",
    }

    us_ttc = sanitize(us101_metrics["ttc_values"], upper=15.0)
    cs_raw_ttc = sanitize(city_raw_metrics["ttc_values"], upper=15.0)
    cs_aligned_ttc = sanitize(city_aligned_metrics["ttc_values"], upper=15.0)
    us_thw = sanitize(us101_metrics["thw_values"], upper=5.0)
    cs_raw_thw = sanitize(city_raw_metrics["thw_values"], upper=5.0)
    cs_aligned_thw = sanitize(city_aligned_metrics["thw_values"], upper=5.0)

    fig, axes = plt.subplots(1, 2, figsize=(14.8, 5.8), constrained_layout=True)

    draw_density(
        axes[0], us_ttc, "US-101", colors["us101"], (0.0, 15.0), linestyle="-", fill=True
    )
    draw_density(
        axes[0],
        cs_raw_ttc,
        "CitySim raw",
        colors["city_raw"],
        (0.0, 15.0),
        linestyle="--",
        fill=False,
    )
    draw_density(
        axes[0],
        cs_aligned_ttc,
        "CitySim aligned",
        colors["city_aligned"],
        (0.0, 15.0),
        linestyle="-",
        fill=False,
    )
    axes[0].set_xlim(0.0, 15.0)
    axes[0].set_title("(a) TTC Distribution", pad=10, weight="semibold")
    axes[0].set_xlabel("Time to Collision (s)")
    axes[0].set_ylabel("Density")

    draw_density(
        axes[1], us_thw, "US-101", colors["us101"], (0.0, 5.0), linestyle="-", fill=True
    )
    draw_density(
        axes[1],
        cs_raw_thw,
        "CitySim raw",
        colors["city_raw"],
        (0.0, 5.0),
        linestyle="--",
        fill=False,
    )
    draw_density(
        axes[1],
        cs_aligned_thw,
        "CitySim aligned",
        colors["city_aligned"],
        (0.0, 5.0),
        linestyle="-",
        fill=False,
    )
    axes[1].set_xlim(0.0, 5.0)
    axes[1].set_title("(b) THW Distribution", pad=10, weight="semibold")
    axes[1].set_xlabel("Time Headway (s)")
    axes[1].set_ylabel("Density")

    for ax in axes:
        ax.grid(True, alpha=0.28)
        ax.legend(loc="upper right", frameon=True)

    return fig


def make_jsonable(summary):
    out = {}
    for key, value in summary.items():
        if isinstance(value, np.ndarray):
            continue
        out[key] = value
    return out


def main():
    args = parse_args()
    configure_plot_style()

    us101_files = get_us101_files(args.max_files)
    citysim_files = get_citysim_raw_files(args.max_files)

    schema_check = compare_schema(us101_files[0], citysim_files[0])
    us101_orientation = summarize_orientation(us101_files, "US-101")
    citysim_orientation = summarize_orientation(citysim_files, "CitySim_raw")

    us101_metrics = collect_ttc_thw(us101_files, "US-101", speed_mode="magnitude")
    citysim_raw_metrics = collect_ttc_thw(citysim_files, "CitySim_raw", speed_mode="signed")
    citysim_aligned_metrics = collect_ttc_thw(
        citysim_files, "CitySim_aligned", speed_mode="magnitude"
    )

    summary = {
        "schema_check": schema_check,
        "orientation": {
            "us101": us101_orientation,
            "citysim_raw": citysim_orientation,
        },
        "ttc_thw_metrics": {
            "us101_reference": make_jsonable(us101_metrics),
            "citysim_raw_signed": make_jsonable(citysim_raw_metrics),
            "citysim_aligned_magnitude": make_jsonable(citysim_aligned_metrics),
        },
        "alignment_rule": {
            "gap_definition": "lead_y - ego_y - 15ft",
            "raw_citysim_longitudinal_speed": "KF_Vel_Y (signed as stored)",
            "aligned_citysim_longitudinal_speed": "abs(KF_Vel_Y)",
            "aligned_lead_speed": "abs(L*_Leading_Vel)",
            "ttc_rule": "gap / (ego_long_speed - lead_long_speed), rel_speed > 0.1",
            "thw_rule": "gap / ego_long_speed, ego_long_speed > 1.0",
        },
    }

    fig = build_figure(us101_metrics, citysim_raw_metrics, citysim_aligned_metrics)

    if not args.no_save:
        os.makedirs(args.output_dir, exist_ok=True)
        summary_path = os.path.join(args.output_dir, "citysim_alignment_summary.json")
        figure_path = os.path.join(args.output_dir, "US101_vs_CitySim_TTC_THW_Aligned.png")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        fig.savefig(figure_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"[OK] Summary saved to: {summary_path}")
        print(f"[OK] Figure saved to: {figure_path}")

    print(json.dumps(summary["orientation"], ensure_ascii=False, indent=2))
    print(json.dumps(summary["ttc_thw_metrics"], ensure_ascii=False, indent=2))

    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
