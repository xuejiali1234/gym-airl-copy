import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from configs.config import Config
from evaluation.failure_case_full_evaluate import (  # noqa: E402
    DETAIL_FIELDS,
    SUMMARY_FIELDS,
    TARGET_MODELS,
    evaluate_single_trajectory,
    load_dataset,
    summarize_rows,
    write_csv,
    write_overlap,
)


DEFAULT_HARD_CASE_DIR = ROOT_DIR / "train_log" / "failure_case_full_eval_20260427_205244"
DEFAULT_HARD_CASE_LIST = DEFAULT_HARD_CASE_DIR / "common15_trajectory_summary_for_gpt.csv"

PROTOCOL_SUMMARY_FIELDS = [
    "model_tag",
    "split",
    "checkpoint",
    "total",
    "merge_success_rate",
    "endpoint_success_rate",
    "safety_success_rate",
    "collision_rate",
    "collision_count",
    "endpoint_fail_count",
    "safety_fail_count",
    "mean_episode_reward",
    "mean_eval_dense_return",
    "mean_paper_score",
    "mean_steps",
    "mean_speed_mps",
    "mean_abs_jerk_mps3",
    "mean_min_ttc",
    "mean_min_thw",
    "ttc_lt_3_rate",
    "ttc_lt_4_rate",
    "thw_lt_0p5_rate",
    "endpoint_gate_pass",
    "safety_gate_pass",
    "collision_gate_pass",
    "hard_zero_collision_pass",
    "protocol_pass",
    "note",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fixed full-set + hard-case evaluation protocol. "
            "This script does not train models and does not change checkpoint saving."
        )
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="TAG=CHECKPOINT",
        help=(
            "Model to evaluate. Can be repeated. "
            "If omitted, the three built-in candidate checkpoints are evaluated."
        ),
    )
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated model tags/prefixes to run from the selected model list.",
    )
    parser.add_argument(
        "--hard-list",
        default=str(DEFAULT_HARD_CASE_LIST),
        help=(
            "CSV containing a filename column for hard cases. Defaults to the common-15 list "
            "created by failure-case analysis."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory. Defaults to train_log/hard_case_protocol_eval_<timestamp>.",
    )
    parser.add_argument(
        "--hard-only",
        action="store_true",
        help="Evaluate only the hard-case split. Useful for quick checks.",
    )
    parser.add_argument("--endpoint-threshold", type=float, default=0.95)
    parser.add_argument("--safety-threshold", type=float, default=0.90)
    parser.add_argument("--collision-threshold", type=float, default=0.01)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected models and hard-case filenames without loading policies.",
    )
    return parser.parse_args()


def parse_model_arg(item):
    if "=" not in item:
        raise ValueError(f"Expected TAG=CHECKPOINT for --model, got: {item!r}")
    tag, checkpoint = item.split("=", 1)
    tag = tag.strip()
    checkpoint = checkpoint.strip().strip('"')
    if not tag or not checkpoint:
        raise ValueError(f"Expected TAG=CHECKPOINT for --model, got: {item!r}")
    return {
        "tag": tag,
        "checkpoint": Path(checkpoint).expanduser().resolve(),
        "note": "User-provided checkpoint.",
    }


def get_models(args):
    if args.model:
        models = [parse_model_arg(item) for item in args.model]
    else:
        models = [
            {
                "tag": item["tag"],
                "checkpoint": Path(item["checkpoint"]).resolve(),
                "note": item.get("note", ""),
            }
            for item in TARGET_MODELS
        ]

    if args.only:
        requested = [part.strip() for part in args.only.split(",") if part.strip()]
        models = [
            item
            for item in models
            if any(item["tag"] == req or item["tag"].startswith(req) for req in requested)
        ]
        if not models:
            raise ValueError(f"No models matched --only={args.only!r}")

    for item in models:
        if not item["checkpoint"].exists():
            raise FileNotFoundError(f"Missing checkpoint for {item['tag']}: {item['checkpoint']}")
    return models


def load_hard_case_filenames(path):
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Missing hard-case list: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "filename" not in reader.fieldnames:
            raise ValueError(f"Hard-case list must contain a filename column: {path}")
        filenames = [row["filename"] for row in reader if row.get("filename")]

    filenames = sorted(dict.fromkeys(filenames))
    if not filenames:
        raise ValueError(f"No filenames found in hard-case list: {path}")
    return path, filenames


def dataset_indices_by_filename(dataset, filenames):
    target = set(filenames)
    matched = []
    missing = set(filenames)
    for idx, traj in enumerate(dataset.trajectories):
        filename = traj.get("filename", f"trajectory_{idx}.csv")
        if filename in target:
            matched.append(idx)
            missing.discard(filename)
    if missing:
        raise ValueError(
            "Hard-case filenames missing from dataset: "
            + ", ".join(sorted(missing))
        )
    return matched


def evaluate_indices(model, dataset, cfg, indices, tag, split_name):
    rows = []
    for traj_index in tqdm(indices, desc=f"{tag}:{split_name}"):
        row = evaluate_single_trajectory(model, dataset, cfg, traj_index)
        row["model_tag"] = tag
        rows.append(row)
    return rows


def annotate_summary(summary, split_name, thresholds):
    endpoint_pass = float(summary["endpoint_success_rate"]) >= thresholds["endpoint"]
    safety_pass = float(summary["safety_success_rate"]) >= thresholds["safety"]
    collision_pass = float(summary["collision_rate"]) <= thresholds["collision"]
    hard_zero_pass = int(summary["collision_count"]) == 0 if split_name == "hard15" else ""
    protocol_pass = endpoint_pass and safety_pass and collision_pass
    if split_name == "hard15":
        protocol_pass = protocol_pass and bool(hard_zero_pass)

    out = {
        key: summary.get(key, "")
        for key in SUMMARY_FIELDS
        if key in summary
    }
    out["split"] = split_name
    out["endpoint_gate_pass"] = endpoint_pass
    out["safety_gate_pass"] = safety_pass
    out["collision_gate_pass"] = collision_pass
    out["hard_zero_collision_pass"] = hard_zero_pass
    out["protocol_pass"] = bool(protocol_pass)
    return {key: out.get(key, "") for key in PROTOCOL_SUMMARY_FIELDS}


def write_split_outputs(output_dir, tag, split_name, rows):
    prefix = f"{tag}_{split_name}"
    write_csv(output_dir / f"{prefix}_trajectory_details.csv", rows, DETAIL_FIELDS)
    write_csv(
        output_dir / f"{prefix}_failure_cases.csv",
        [row for row in rows if row["failure_type"] != "success"],
        DETAIL_FIELDS,
    )
    write_csv(
        output_dir / f"{prefix}_collision_cases.csv",
        [row for row in rows if row["collision"]],
        DETAIL_FIELDS,
    )


def fmt_rate(value):
    return f"{float(value):.3f}"


def write_protocol_report(output_dir, summaries, hard_filenames, thresholds):
    report = [
        "# Fixed hard-case evaluation protocol",
        "",
        f"Created at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Gates",
        "",
        f"- Full endpoint_success_rate >= {thresholds['endpoint']}",
        f"- Full safety_success_rate >= {thresholds['safety']}",
        f"- Full collision_rate <= {thresholds['collision']}",
        "- Hard-15 collision_count == 0 is reported as an additional hard-case target.",
        "",
        "## Hard-case list",
        "",
        f"Hard cases: {len(hard_filenames)} trajectories.",
        "",
        "```text",
        *hard_filenames,
        "```",
        "",
        "## Summary",
        "",
        "| Model | Split | N | Endpoint | Safety | Collision | Collisions | Mean TTC | Mean THW | Pass |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    for row in summaries:
        report.append(
            "| {model} | {split} | {total} | {endpoint} | {safety} | {collision} | "
            "{collision_count} | {ttc} | {thw} | {passed} |".format(
                model=row["model_tag"],
                split=row["split"],
                total=row["total"],
                endpoint=fmt_rate(row["endpoint_success_rate"]),
                safety=fmt_rate(row["safety_success_rate"]),
                collision=fmt_rate(row["collision_rate"]),
                collision_count=row["collision_count"],
                ttc=fmt_rate(row["mean_min_ttc"]),
                thw=fmt_rate(row["mean_min_thw"]),
                passed=row["protocol_pass"],
            )
        )

    report.extend(
        [
            "",
            "## Notes",
            "",
            "This script is evaluation-only. It does not modify training rewards, checkpoints, "
            "the environment, or model-selection logic in train_airl_baseline.py.",
            "",
            "Use the full split to check general performance, and the hard15 split to check "
            "whether the model handles the previously identified common collision cases.",
            "",
        ]
    )
    (output_dir / "protocol_report.md").write_text("\n".join(report), encoding="utf-8-sig")


def main():
    args = parse_args()
    models = get_models(args)
    hard_list_path, hard_filenames = load_hard_case_filenames(args.hard_list)

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else ROOT_DIR / "train_log" / f"hard_case_protocol_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    thresholds = {
        "endpoint": args.endpoint_threshold,
        "safety": args.safety_threshold,
        "collision": args.collision_threshold,
    }

    print("=" * 80)
    print("Fixed hard-case evaluation protocol")
    print(f"Output: {output_dir}")
    print(f"Hard-case list: {hard_list_path}")
    print(f"Hard cases: {len(hard_filenames)}")
    print("=" * 80)
    for item in models:
        print(f"- {item['tag']}: {item['checkpoint']}")

    if args.dry_run:
        print("[Dry run] No policy loaded and no evaluation executed.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()
    dataset = load_dataset(device)
    hard_indices = dataset_indices_by_filename(dataset, hard_filenames)
    full_indices = list(range(len(dataset)))

    write_csv(
        output_dir / "hard_case_list_used.csv",
        [{"filename": name} for name in hard_filenames],
        ["filename"],
    )
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "collision_margin": 1.0,
        "dataset_size": len(dataset),
        "hard_case_count": len(hard_indices),
        "hard_case_list": str(hard_list_path),
        "thresholds": thresholds,
        "hard_only": args.hard_only,
        "models": [
            {
                "tag": item["tag"],
                "checkpoint": str(item["checkpoint"]),
                "note": item.get("note", ""),
            }
            for item in models
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8-sig",
    )

    protocol_summaries = []
    full_rows_by_model = {}
    hard_rows_by_model = {}
    splits = [("hard15", hard_indices)] if args.hard_only else [
        ("full", full_indices),
        ("hard15", hard_indices),
    ]

    for model_info in models:
        tag = model_info["tag"]
        print("\n" + "-" * 80)
        print(f"Loading {tag}")
        model = PPO.load(str(model_info["checkpoint"]), device=device)

        for split_name, indices in splits:
            rows = evaluate_indices(model, dataset, cfg, indices, tag, split_name)
            write_split_outputs(output_dir, tag, split_name, rows)

            summary = summarize_rows(model_info, rows)
            annotated = annotate_summary(summary, split_name, thresholds)
            protocol_summaries.append(annotated)
            if split_name == "full":
                full_rows_by_model[tag] = rows
            elif split_name == "hard15":
                hard_rows_by_model[tag] = rows

            print(
                f"{tag}/{split_name}: N={annotated['total']}, "
                f"endpoint={float(annotated['endpoint_success_rate']):.3f}, "
                f"safety={float(annotated['safety_success_rate']):.3f}, "
                f"collision={float(annotated['collision_rate']):.3f}, "
                f"collisions={annotated['collision_count']}, "
                f"pass={annotated['protocol_pass']}"
            )

    write_csv(output_dir / "protocol_summary.csv", protocol_summaries, PROTOCOL_SUMMARY_FIELDS)
    if full_rows_by_model:
        write_overlap(output_dir / "full_overlap", full_rows_by_model)
    if hard_rows_by_model:
        write_overlap(output_dir / "hard15_overlap", hard_rows_by_model)
    write_protocol_report(output_dir, protocol_summaries, hard_filenames, thresholds)

    print("\n" + "=" * 80)
    print(f"Saved fixed protocol evaluation to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
