#!/usr/bin/env python3
"""
Export a persistent CitySim aligned dataset without overwriting the original
raw files.

Alignment scope
---------------
This exporter only fixes the longitudinal kinematics used by the current
analysis / loader stack:

- KF_Vel_Y                -> abs(KF_Vel_Y)
- KF_Acc_Y                -> same file-level sign alignment as KF_Vel_Y
- KF_Vel / KF_Acc         -> kept magnitude-consistent
- v_Vel                   -> kept as raw magnitude-like speed
- v_Acc                   -> same file-level sign alignment as KF_Vel_Y
- L*_Vel                  -> abs(L*_Vel)
- L*_Acc                  -> local sign alignment where raw L*_Vel is negative

Not changed
-----------
- Local_X / Local_Y / Global_X / Global_Y coordinates
- Lane_ID semantics (CitySim raw still uses lane ids such as 0/1)
- Preceeding / Following / Space_Hdwy / Time_Hdwy
- *_normalized folders are not exported

Outputs
-------
- data-CitySim-aligned/lane_change_trajectories-FreewayC-*/
- data-CitySim-aligned/alignment_manifest.json
"""

import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.join(ROOT_DIR, "data-CitySim")
DEFAULT_OUTPUT_ROOT = os.path.join(ROOT_DIR, "data-CitySim-aligned")
DT = 0.1
CAR_LEN_FT = 15.0


SURROUND_PREFIXES = [
    "L6_Leading",
    "L5_Leading",
    "L5_Following",
    "L6_Following",
]


@dataclass
class ExportSummary:
    files_written: int = 0
    rows_written: int = 0
    raw_vy_negative: int = 0
    aligned_vy_positive: int = 0
    raw_candidate_rows: int = 0
    raw_positive_rel_speed_rows: int = 0
    aligned_candidate_rows: int = 0
    aligned_positive_rel_speed_rows: int = 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a persistent longitudinally aligned CitySim dataset."
    )
    parser.add_argument(
        "--source-root",
        default=SRC_ROOT,
        help="Path to raw CitySim root containing lane_change_trajectories-FreewayC-* dirs.",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory to save the aligned dataset.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional file cap for quick smoke export.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output CSV files.",
    )
    return parser.parse_args()


def find_source_dirs(source_root):
    dirs = sorted(glob.glob(os.path.join(source_root, "lane_change_trajectories-FreewayC-*")))
    dirs = [d for d in dirs if os.path.isdir(d) and not d.endswith("_normalized")]
    if not dirs:
        raise FileNotFoundError(f"No raw CitySim directories found under: {source_root}")
    return dirs


def compute_magnitude(vx, vy):
    return np.sqrt(np.asarray(vx, dtype=float) ** 2 + np.asarray(vy, dtype=float) ** 2)


def valid_surround_mask(df, prefix):
    cols = [
        f"{prefix}_Local_X",
        f"{prefix}_Local_Y",
        f"{prefix}_Vel",
        f"{prefix}_Acc",
    ]
    present = [col for col in cols if col in df.columns]
    if not present:
        return np.zeros(len(df), dtype=bool)

    accum = np.zeros(len(df), dtype=bool)
    for col in present:
        values = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy()
        accum |= np.abs(values) > 1e-9
    return accum


def infer_ego_direction_sign(df):
    raw_vy = pd.to_numeric(df["KF_Vel_Y"], errors="coerce").to_numpy()
    finite = raw_vy[np.isfinite(raw_vy)]
    if finite.size == 0:
        return 1.0
    return -1.0 if np.median(finite) < 0 else 1.0


def align_dataframe(df):
    aligned = df.copy()
    ego_sign = infer_ego_direction_sign(df)

    # Ego longitudinal kinematics
    raw_kf_vy = pd.to_numeric(aligned["KF_Vel_Y"], errors="coerce").to_numpy(dtype=float)
    raw_kf_ay = pd.to_numeric(aligned["KF_Acc_Y"], errors="coerce").to_numpy(dtype=float)
    aligned["KF_Vel_Y"] = np.abs(raw_kf_vy)
    aligned["KF_Acc_Y"] = ego_sign * raw_kf_ay
    aligned["KF_Vel"] = compute_magnitude(
        pd.to_numeric(aligned["KF_Vel_X"], errors="coerce").to_numpy(dtype=float),
        aligned["KF_Vel_Y"].to_numpy(dtype=float),
    )
    aligned["KF_Acc"] = compute_magnitude(
        pd.to_numeric(aligned["KF_Acc_X"], errors="coerce").to_numpy(dtype=float),
        aligned["KF_Acc_Y"].to_numpy(dtype=float),
    )

    if "v_Acc" in aligned.columns:
        raw_v_acc = pd.to_numeric(aligned["v_Acc"], errors="coerce").to_numpy(dtype=float)
        aligned["v_Acc"] = ego_sign * raw_v_acc

    # Surround vehicle longitudinal kinematics
    for prefix in SURROUND_PREFIXES:
        vel_col = f"{prefix}_Vel"
        acc_col = f"{prefix}_Acc"
        if vel_col not in aligned.columns:
            continue

        raw_vel = pd.to_numeric(aligned[vel_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        valid_mask = valid_surround_mask(aligned, prefix)
        aligned_vel = np.abs(raw_vel)
        aligned_vel[~valid_mask] = 0.0
        aligned[vel_col] = aligned_vel

        if acc_col in aligned.columns:
            raw_acc = pd.to_numeric(aligned[acc_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            aligned_acc = raw_acc.copy()
            aligned_acc[raw_vel < 0] *= -1.0
            aligned_acc[~valid_mask] = 0.0
            aligned[acc_col] = aligned_acc

    return aligned


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


def collect_rel_speed_stats(df, speed_mode):
    py = pd.to_numeric(df["KF_Local_Y"], errors="coerce").to_numpy()
    vy = pd.to_numeric(df["KF_Vel_Y"], errors="coerce").to_numpy()
    l6_y = pd.to_numeric(df["L6_Leading_Local_Y"], errors="coerce").to_numpy()
    l6_v = pd.to_numeric(df["L6_Leading_Vel"], errors="coerce").to_numpy()
    l5_y = pd.to_numeric(df["L5_Leading_Local_Y"], errors="coerce").to_numpy()
    l5_v = pd.to_numeric(df["L5_Leading_Vel"], errors="coerce").to_numpy()

    candidate_rows = 0
    positive_rel_speed_rows = 0

    for i in range(len(df)):
        gap, lead_v = select_nearest_lead_gap(py[i], l6_y[i], l6_v[i], l5_y[i], l5_v[i])
        if gap is None:
            continue
        candidate_rows += 1

        if speed_mode == "signed":
            ego_long = vy[i]
            lead_long = lead_v
        elif speed_mode == "magnitude":
            ego_long = abs(vy[i])
            lead_long = abs(lead_v)
        else:
            raise ValueError(f"Unknown speed_mode: {speed_mode}")

        if (ego_long - lead_long) > 0.1:
            positive_rel_speed_rows += 1

    return candidate_rows, positive_rel_speed_rows


def source_schema_summary(sample_csv):
    df = pd.read_csv(sample_csv, nrows=5)
    return {
        "column_count": len(df.columns),
        "columns": list(df.columns),
        "dtypes": {k: str(v) for k, v in df.dtypes.items()},
    }


def export_dataset(args):
    source_dirs = find_source_dirs(args.source_root)
    os.makedirs(args.output_root, exist_ok=True)

    summary = ExportSummary()
    sample_input_csv = None
    sample_output_csv = None
    directory_summaries = []

    file_budget = args.max_files
    stop_early = False

    for src_dir in source_dirs:
        rel_dir = os.path.basename(src_dir)
        dst_dir = os.path.join(args.output_root, rel_dir)
        os.makedirs(dst_dir, exist_ok=True)

        csv_files = sorted(glob.glob(os.path.join(src_dir, "vehicle_*.csv")))
        written_here = 0

        for src_csv in csv_files:
            if file_budget is not None and summary.files_written >= file_budget:
                stop_early = True
                break

            filename = os.path.basename(src_csv)
            dst_csv = os.path.join(dst_dir, filename)
            if os.path.exists(dst_csv) and not args.overwrite:
                raise FileExistsError(
                    f"Output file already exists: {dst_csv}\n"
                    f"Use --overwrite or choose a different --output-root."
                )

            raw_df = pd.read_csv(src_csv)
            aligned_df = align_dataframe(raw_df)

            summary.files_written += 1
            summary.rows_written += len(raw_df)
            summary.raw_vy_negative += int(
                np.sum(pd.to_numeric(raw_df["KF_Vel_Y"], errors="coerce").to_numpy() < -1e-6)
            )
            summary.aligned_vy_positive += int(
                np.sum(pd.to_numeric(aligned_df["KF_Vel_Y"], errors="coerce").to_numpy() > 1e-6)
            )

            raw_cand, raw_rel = collect_rel_speed_stats(raw_df, speed_mode="signed")
            aligned_cand, aligned_rel = collect_rel_speed_stats(aligned_df, speed_mode="signed")
            summary.raw_candidate_rows += raw_cand
            summary.raw_positive_rel_speed_rows += raw_rel
            summary.aligned_candidate_rows += aligned_cand
            summary.aligned_positive_rel_speed_rows += aligned_rel

            aligned_df.to_csv(dst_csv, index=False)
            written_here += 1

            if sample_input_csv is None:
                sample_input_csv = src_csv
                sample_output_csv = dst_csv

        directory_summaries.append(
            {
                "source_dir": src_dir,
                "output_dir": dst_dir,
                "files_written": written_here,
            }
        )

        if stop_early:
            break

    if sample_input_csv is None or sample_output_csv is None:
        raise RuntimeError("No CitySim files were exported.")

    manifest = {
        "source_root": args.source_root,
        "output_root": args.output_root,
        "files_written": summary.files_written,
        "rows_written": summary.rows_written,
        "alignment_scope": {
            "kept_untouched": [
                "KF_Local_X",
                "KF_Local_Y",
                "Local_X",
                "Local_Y",
                "Global_X",
                "Global_Y",
                "Lane_ID",
                "Preceeding",
                "Following",
                "Space_Hdwy",
                "Time_Hdwy",
            ],
            "aligned_columns": {
                "KF_Vel_Y": "abs(KF_Vel_Y)",
                "KF_Acc_Y": "file-level sign alignment matched to KF_Vel_Y",
                "KF_Vel": "recomputed magnitude from KF_Vel_X / aligned KF_Vel_Y",
                "KF_Acc": "recomputed magnitude from KF_Acc_X / aligned KF_Acc_Y",
                "v_Vel": "kept raw positive magnitude-like speed",
                "v_Acc": "file-level sign alignment matched to KF_Vel_Y",
                "L*_Vel": "abs(L*_Vel) on valid segments, zero on missing segments",
                "L*_Acc": "sign-flipped only where raw L*_Vel < 0, zero on missing segments",
            },
        },
        "known_limitations": [
            "KF_Local_Y / Local_Y / Global_Y are not mirrored or re-anchored.",
            "Lane_ID semantics are preserved as raw CitySim ids (for example 0/1), not remapped to US-101 style lane ids.",
            "This aligned dataset fixes longitudinal kinematics for current TTC/THW analysis and loader-side speed semantics, but it is not yet a full coordinate-system harmonization for drop-in training with the current US-101-style environment dynamics.",
        ],
        "validation": {
            "raw_vy_negative_rate": round(
                summary.raw_vy_negative / max(summary.rows_written, 1), 4
            ),
            "aligned_vy_positive_rate": round(
                summary.aligned_vy_positive / max(summary.rows_written, 1), 4
            ),
            "raw_ttc_positive_rel_speed_rate_candidate_rows": round(
                summary.raw_positive_rel_speed_rows / max(summary.raw_candidate_rows, 1), 4
            ),
            "aligned_ttc_positive_rel_speed_rate_candidate_rows": round(
                summary.aligned_positive_rel_speed_rows / max(summary.aligned_candidate_rows, 1), 4
            ),
        },
        "source_schema_sample": source_schema_summary(sample_input_csv),
        "sample_files": {
            "input": sample_input_csv,
            "output": sample_output_csv,
        },
        "directory_summaries": directory_summaries,
    }

    manifest_path = os.path.join(args.output_root, "alignment_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return manifest_path, manifest


def main():
    args = parse_args()
    manifest_path, manifest = export_dataset(args)
    print(f"[OK] Aligned dataset exported to: {manifest['output_root']}")
    print(f"[OK] Manifest saved to: {manifest_path}")
    print(json.dumps(manifest["validation"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
