from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator


ROOT_DIR = Path(__file__).resolve().parents[1]
P30_DIR = (
    ROOT_DIR
    / "train_log"
    / "baseline_attn_goal_safe_branch_aux_probe_P30_CPairD250_NoLateLR_Save1_20260502_215110"
)
REEVAL_DIR = ROOT_DIR / "train_log" / "ReEval_P30_KeyCkpts_20260503"
OUT_DIR = ROOT_DIR / "plot" / "p30_paper_story"

METRICS = [
    ("merge_success_rate", "Merge", "Merge rate", (0.0, 1.05), "#0072B2"),
    ("endpoint_success_rate", "Endpoint", "Endpoint rate", (0.0, 1.05), "#009E73"),
    ("safety_success_rate", "Safety", "Safety rate", (0.0, 1.05), "#6A51A3"),
    ("collision_rate", "Collision", "Collision rate", (0.0, 0.95), "#D55E00"),
]
KEY_EPOCHS = [260, 270, 280, 292, 295, 298, 300]
SELECT_EPOCHS = [292, 295, 298]


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "figure.dpi": 160,
            "savefig.dpi": 450,
            "axes.linewidth": 0.85,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_training_metrics() -> pd.DataFrame:
    path = P30_DIR / "eval_metrics.csv"
    df = pd.read_csv(path)
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["epoch"]).copy()
    df["epoch"] = df["epoch"].astype(int)
    for column, *_ in METRICS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.sort_values("epoch").reset_index(drop=True)


def load_protocol_summary() -> pd.DataFrame:
    path = REEVAL_DIR / "protocol_summary.csv"
    df = pd.read_csv(path)
    keep = df["model_tag"].isin([f"P30_CPairD250_{epoch}" for epoch in SELECT_EPOCHS])
    df = df[keep].copy()
    df["epoch"] = df["model_tag"].str.extract(r"_(\d+)$").astype(int)
    return df.sort_values(["split", "epoch"]).reset_index(drop=True)


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=1).mean()


def draw_late_background(ax: plt.Axes) -> None:
    ax.axvspan(220, 230, color="#FFF4D6", alpha=0.75, lw=0)
    ax.axvspan(230, 250, color="#E8F4EA", alpha=0.75, lw=0)
    ax.axvspan(250, 300, color="#F8E8E8", alpha=0.75, lw=0)
    ax.axvspan(292, 298, color="#2A9D8F", alpha=0.14, lw=0)
    for epoch in [220, 230, 250, 292, 298, 300]:
        ax.axvline(epoch, color="#666666", lw=0.70, ls=":", alpha=0.55)


def polish_axis(ax: plt.Axes, ylim: tuple[float, float], ylabel: str) -> None:
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid(axis="y", color="#D9D9D9", lw=0.65, alpha=0.78)
    ax.grid(axis="x", color="#EEEEEE", lw=0.45, alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", length=3.0, width=0.75)


def save(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = OUT_DIR / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        print(f"saved: {path}")
    plt.close(fig)


def plot_late_window(df: pd.DataFrame) -> None:
    late = df[(df["epoch"] >= 220) & (df["epoch"] <= 300)].copy()
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.4), sharex=True)

    for idx, (ax, (column, title, ylabel, ylim, color)) in enumerate(zip(axes.ravel(), METRICS)):
        draw_late_background(ax)
        ax.plot(late["epoch"], late[column], color=color, alpha=0.22, lw=0.9, label="Raw")
        ax.plot(late["epoch"], ema(late[column], 5), color=color, lw=2.2, label="EMA-5")
        selected = late[late["epoch"].isin(SELECT_EPOCHS)]
        ax.scatter(
            selected["epoch"],
            selected[column],
            s=30,
            facecolor="white",
            edgecolor=color,
            linewidth=1.0,
            zorder=4,
        )
        ax.set_title(f"({chr(97 + idx)}) {title}", loc="left", fontsize=10, fontweight="bold")
        polish_axis(ax, ylim, ylabel)

    for ax in axes[-1, :]:
        ax.set_xlabel("Epoch")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.52, 1.00))
    fig.text(
        0.5,
        0.955,
        "Late-stage training window: U220, D230 and Decay250 are shown as shaded phases",
        ha="center",
        va="top",
        fontsize=8.2,
        color="#444444",
    )
    fig.tight_layout(rect=(0.02, 0.02, 1.0, 0.91), w_pad=1.2, h_pad=1.2)
    save(fig, "p30_late_window_raw_ema")


def plot_checkpoint_selection(df: pd.DataFrame) -> None:
    key = df[df["epoch"].isin(KEY_EPOCHS)].copy()
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    x = list(range(len(key)))
    x_labels = [f"@{epoch}" for epoch in key["epoch"]]

    lines = [
        ("merge_success_rate", "Merge", "#0072B2", "-"),
        ("endpoint_success_rate", "Endpoint", "#009E73", "-"),
        ("safety_success_rate", "Safety", "#6A51A3", "-"),
        ("collision_rate", "Collision", "#D55E00", "--"),
    ]
    for column, label, color, linestyle in lines:
        ax.plot(
            x,
            key[column],
            marker="o",
            ms=5.5,
            lw=2.0,
            ls=linestyle,
            color=color,
            label=label,
        )

    ax.axvspan(2.5, 5.5, color="#2A9D8F", alpha=0.12, lw=0)
    ax.axvline(5, color="#333333", lw=0.9, ls=":", alpha=0.75)
    ax.text(5, 1.015, "Selected @298", ha="center", va="bottom", fontsize=8.2, color="#333333")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Checkpoint epoch")
    ax.set_ylabel("Rate")
    ax.set_xticks(x, x_labels)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(axis="y", color="#D9D9D9", lw=0.65, alpha=0.78)
    ax.grid(axis="x", color="#EEEEEE", lw=0.45, alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower left", ncol=2, frameon=False)
    fig.tight_layout()
    save(fig, "p30_key_checkpoint_selection")


def plot_fixed_protocol(protocol: pd.DataFrame) -> None:
    full = protocol[protocol["split"] == "full"].copy()
    hard = protocol[protocol["split"] == "hard15"].copy()
    x_labels = [f"@{epoch}" for epoch in SELECT_EPOCHS]

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))

    full = full.set_index("epoch").loc[SELECT_EPOCHS].reset_index()
    axes[0].bar(x_labels, full["collision_count"], color="#D55E00", alpha=0.86, width=0.56)
    axes[0].set_title("(a) full217 collision", loc="left", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Collision count")
    axes[0].text(2, full.loc[full["epoch"] == 298, "collision_count"].iloc[0] + 1.0, "lowest", ha="center", fontsize=8, color="#333333")

    for column, label, color in [
        ("merge_success_rate", "Merge", "#0072B2"),
        ("endpoint_success_rate", "Endpoint", "#009E73"),
        ("safety_success_rate", "Safety", "#6A51A3"),
    ]:
        axes[1].plot(x_labels, full[column], marker="o", lw=2.0, color=color, label=label)
    axes[1].set_title("(b) full217 task rates", loc="left", fontsize=10, fontweight="bold")
    axes[1].set_ylabel("Rate")
    axes[1].set_ylim(0.70, 1.02)
    axes[1].legend(loc="lower right", frameon=False)

    for ax in axes:
        ax.grid(axis="y", color="#D9D9D9", lw=0.65, alpha=0.78)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    hard = hard.set_index("epoch").loc[SELECT_EPOCHS].reset_index()
    hard_counts = ", ".join(f"@{row.epoch}: {int(row.collision_count)}/15" for row in hard.itertuples())
    fig.text(0.5, -0.02, f"hard15 collision: {hard_counts}", ha="center", va="top", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "p30_fixed_protocol_selection")


def main() -> None:
    configure_style()
    training = load_training_metrics()
    protocol = load_protocol_summary()
    plot_late_window(training)
    plot_checkpoint_selection(training)
    plot_fixed_protocol(protocol)


if __name__ == "__main__":
    main()
