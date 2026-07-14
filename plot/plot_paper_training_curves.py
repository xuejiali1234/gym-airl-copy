from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator


ROOT_DIR = Path(__file__).resolve().parents[1]

DEFAULT_P30_DIR = (
    ROOT_DIR
    / "train_log"
    / "baseline_attn_goal_safe_branch_aux_probe_P30_CPairD250_NoLateLR_Save1_20260502_215110"
)
DEFAULT_P30_RERUN_DIR = (
    ROOT_DIR
    / "train_log"
    / "baseline_attn_goal_safe_branch_aux_probe_P30_CPairD250_NoLateLR_Save1_Rerun_20260506_131051"
)
DEFAULT_P45_DIR = (
    ROOT_DIR
    / "train_log"
    / "baseline_attn_goal_safe_branch_aux_probe_P45_P30_SmoothTransitions_20260506_114332"
)
DEFAULT_OUTPUT_DIR = ROOT_DIR / "plot" / "paper_training_curves"


METRICS = [
    ("merge_success_rate", "(a) Merge", "Merge rate", (0.0, 1.05), "#0072B2"),
    ("endpoint_success_rate", "(b) Endpoint", "Endpoint rate", (0.0, 1.05), "#009E73"),
    ("safety_success_rate", "(c) Safety", "Safety rate", (0.0, 1.05), "#6A51A3"),
    ("collision_rate", "(d) Collision", "Collision rate", (0.0, 0.95), "#D55E00"),
]

PHASES = [
    (0, 220, "Base", "#F5F7FA"),
    (220, 230, "U220", "#FFF4D6"),
    (230, 250, "D230", "#E8F4EA"),
    (250, 300, "Decay250", "#F8E8E8"),
]

KEY_EPOCHS = [292, 298, 300]
SWEET_WINDOW = (292, 298)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "figure.dpi": 160,
            "savefig.dpi": 450,
            "axes.linewidth": 0.8,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def resolve_metrics_csv(path: Path) -> Path:
    if path.is_file():
        return path
    csv_path = path / "eval_metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Cannot find eval_metrics.csv under: {path}")
    return csv_path


def load_metrics(path: Path, start_epoch: int | None, end_epoch: int | None) -> pd.DataFrame:
    csv_path = resolve_metrics_csv(path)
    df = pd.read_csv(csv_path)
    if "epoch" not in df.columns:
        raise ValueError(f"{csv_path} does not contain an epoch column")

    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    df = df.dropna(subset=["epoch"]).copy()
    df["epoch"] = df["epoch"].astype(int)
    df = df.sort_values("epoch")

    for column, *_ in METRICS:
        if column not in df.columns:
            raise ValueError(f"{csv_path} does not contain required column: {column}")
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if start_epoch is not None:
        df = df[df["epoch"] >= start_epoch]
    if end_epoch is not None:
        df = df[df["epoch"] <= end_epoch]

    return df.reset_index(drop=True)


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=1).mean()


def draw_phase_background(ax: plt.Axes, show_labels: bool, x_max: int) -> None:
    for start, end, _, color in PHASES:
        ax.axvspan(start, end, color=color, alpha=0.70, lw=0, zorder=0)

    if x_max > 300:
        ax.axvspan(300, x_max, color="#EEF3F8", alpha=0.80, lw=0, zorder=0)

    if show_labels:
        compact_labels = [
            (110, "Base"),
            (235, "Update"),
            (275, "Decay250"),
        ]
        if x_max > 300:
            compact_labels.append(((300 + x_max) / 2, "Post300"))
        for x_pos, label in compact_labels:
            ax.text(
                x_pos,
                1.035,
                label,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=9,
                color="#444444",
            )

    ax.axvspan(
        SWEET_WINDOW[0],
        SWEET_WINDOW[1],
        color="#2A9D8F",
        alpha=0.13,
        lw=0,
        zorder=0,
    )
    if show_labels:
        ax.text(
            sum(SWEET_WINDOW) / 2,
            0.955,
            "292-298 sweet window",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9,
            color="#16695D",
        )

    for epoch in KEY_EPOCHS:
        ax.axvline(epoch, color="#555555", lw=0.75, ls=":", alpha=0.60, zorder=1)


def polish_axis(ax: plt.Axes, ylim: tuple[float, float], ylabel: str) -> None:
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid(axis="y", color="#D9D9D9", lw=0.65, alpha=0.80)
    ax.grid(axis="x", color="#EEEEEE", lw=0.45, alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", length=3.0, width=0.75, color="#333333")


def plot_main_p30(
    df: pd.DataFrame,
    out_dir: Path,
    ema_span: int,
    formats: Iterable[str],
    stem: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.2), sharex=True)
    axes_flat = axes.ravel()
    x_max = int(df["epoch"].max())

    for idx, (ax, (column, title, ylabel, ylim, color)) in enumerate(zip(axes_flat, METRICS)):
        draw_phase_background(ax, show_labels=(idx == 0), x_max=x_max)
        ax.plot(
            df["epoch"],
            df[column],
            color=color,
            alpha=0.20,
            lw=0.85,
            label="Raw",
            zorder=2,
        )
        ax.plot(
            df["epoch"],
            ema(df[column], ema_span),
            color=color,
            alpha=0.98,
            lw=2.15,
            label=f"EMA-{ema_span}",
            zorder=3,
        )

        key_rows = df[df["epoch"].isin(KEY_EPOCHS)]
        ax.scatter(
            key_rows["epoch"],
            key_rows[column],
            s=26,
            facecolor="white",
            edgecolor=color,
            linewidth=1.0,
            zorder=4,
        )
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=7)
        polish_axis(ax, ylim, ylabel)

    for ax in axes[-1, :]:
        ax.set_xlabel("Epoch")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.52, 1.005),
        frameon=False,
    )
    fig.tight_layout(rect=(0.02, 0.02, 1.0, 0.95), w_pad=1.5, h_pad=1.6)
    save_figure(fig, out_dir, stem, formats)


def plot_compare_runs(
    runs: list[tuple[str, pd.DataFrame, str, str]],
    out_dir: Path,
    ema_span: int,
    formats: Iterable[str],
    stem: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.2), sharex=True)
    axes_flat = axes.ravel()
    x_max = max(int(df["epoch"].max()) for _, df, _, _ in runs)

    for idx, (ax, (column, title, ylabel, ylim, _)) in enumerate(zip(axes_flat, METRICS)):
        draw_phase_background(ax, show_labels=(idx == 0), x_max=x_max)
        for name, df, color, linestyle in runs:
            ax.plot(
                df["epoch"],
                ema(df[column], ema_span),
                color=color,
                ls=linestyle,
                lw=2.05,
                alpha=0.95,
                label=name,
                zorder=3,
            )
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=7)
        polish_axis(ax, ylim, ylabel)

    for ax in axes[-1, :]:
        ax.set_xlabel("Epoch")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=min(len(runs), 3),
        bbox_to_anchor=(0.52, 1.005),
        frameon=False,
    )
    fig.tight_layout(rect=(0.02, 0.02, 1.0, 0.95), w_pad=1.5, h_pad=1.6)
    save_figure(fig, out_dir, stem, formats)


def save_figure(fig: plt.Figure, out_dir: Path, stem: str, formats: Iterable[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt.lower()}"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        print(f"saved: {path}")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Draw paper-style AIRL probe curves using phase bands and EMA smoothing. "
            "This script only visualizes existing eval_metrics.csv files; it does not retrain."
        )
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_P30_DIR)
    parser.add_argument("--rerun-dir", type=Path, default=DEFAULT_P30_RERUN_DIR)
    parser.add_argument("--p45-dir", type=Path, default=DEFAULT_P45_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ema-span", type=int, default=9)
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--end-epoch", type=int, default=300)
    parser.add_argument("--run-label", default="P30 original")
    parser.add_argument("--rerun-label", default="P30 rerun")
    parser.add_argument("--main-stem", default="p30_training_phased_smoothed")
    parser.add_argument("--compare-stem", default="p30_original_vs_rerun")
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf", "svg"])
    parser.add_argument("--skip-diagnostics", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()

    p30 = load_metrics(args.run_dir, args.start_epoch, args.end_epoch)
    plot_main_p30(p30, args.output_dir, args.ema_span, args.formats, args.main_stem)

    if args.skip_diagnostics:
        return

    if args.rerun_dir.exists():
        rerun = load_metrics(args.rerun_dir, args.start_epoch, args.end_epoch)
        plot_compare_runs(
            [
                (args.run_label, p30, "#0072B2", "-"),
                (args.rerun_label, rerun, "#D55E00", "--"),
            ],
            args.output_dir,
            args.ema_span,
            args.formats,
            args.compare_stem,
        )

    if args.p45_dir.exists():
        p45 = load_metrics(args.p45_dir, args.start_epoch, args.end_epoch)
        plot_compare_runs(
            [
                ("P30", p30, "#0072B2", "-"),
                ("P45 smoothing", p45, "#D55E00", "--"),
            ],
            args.output_dir,
            args.ema_span,
            args.formats,
            "p30_vs_p45_smoothing_diagnostic",
        )


if __name__ == "__main__":
    main()
