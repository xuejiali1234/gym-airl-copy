import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
from stable_baselines3 import PPO


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from configs.config import Config
from evaluation.failure_case_full_evaluate import load_dataset, summarize_rows, write_csv, write_overlap
from evaluation.hard_case_protocol_evaluate import (
    PROTOCOL_SUMMARY_FIELDS,
    annotate_summary,
    dataset_indices_by_filename,
    evaluate_indices,
    load_hard_case_filenames,
    write_protocol_report,
    write_split_outputs,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run fixed protocol evaluation for PPO-RL baseline checkpoints.")
    parser.add_argument("--model", action="append", default=[], metavar="TAG=CHECKPOINT")
    parser.add_argument("--checkpoint", default="", help="Single checkpoint path.")
    parser.add_argument("--tag", default="", help="Tag for --checkpoint.")
    parser.add_argument("--only", default="", help="Comma-separated model tags/prefixes to run.")
    parser.add_argument("--hard-list", default="", help="Optional hard-case CSV path.")
    parser.add_argument("--output-dir", default="", help="Output directory.")
    parser.add_argument("--hard-only", action="store_true")
    parser.add_argument("--endpoint-threshold", type=float, default=0.95)
    parser.add_argument("--safety-threshold", type=float, default=0.90)
    parser.add_argument("--collision-threshold", type=float, default=0.01)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_model_arg(item):
    if "=" not in item:
        raise ValueError(f"Expected TAG=CHECKPOINT for --model, got {item!r}")
    tag, checkpoint = item.split("=", 1)
    return {
        "tag": tag.strip(),
        "checkpoint": Path(checkpoint.strip().strip('"')).expanduser().resolve(),
    }


def get_models(args):
    models = [parse_model_arg(item) for item in args.model]
    if args.checkpoint:
        if not args.tag:
            raise ValueError("--tag is required when using --checkpoint.")
        models.append(
            {
                "tag": args.tag.strip(),
                "checkpoint": Path(args.checkpoint).expanduser().resolve(),
            }
        )
    if not models:
        raise ValueError("Provide at least one checkpoint via --model or --checkpoint/--tag.")

    if args.only:
        requested = [part.strip() for part in args.only.split(",") if part.strip()]
        models = [
            item
            for item in models
            if any(item["tag"] == req or item["tag"].startswith(req) for req in requested)
        ]
        if not models:
            raise ValueError(f"No checkpoints matched --only={args.only!r}")

    for item in models:
        if not item["checkpoint"].exists():
            raise FileNotFoundError(f"Missing checkpoint: {item['checkpoint']}")
    return models


def load_run_config(checkpoint_path):
    checkpoint_path = Path(checkpoint_path).resolve()
    run_dir = checkpoint_path.parents[1]
    run_config_path = run_dir / "run_config.json"
    if not run_config_path.exists():
        return {}, run_dir
    with run_config_path.open("r", encoding="utf-8") as f:
        return json.load(f), run_dir


def cfg_for_model(run_config):
    cfg = Config()
    cfg.ENABLE_GOAL_CONDITION = bool(run_config.get("ppo_rl_use_goal", True))
    cfg.GOAL_ABLATION_MODE = "normal"
    cfg.ENABLE_ATTENTION = bool(run_config.get("ppo_rl_use_attention", False))
    if "seed" in run_config:
        cfg.SEED = int(run_config["seed"])
    if isinstance(run_config.get("config"), dict):
        nested = run_config["config"]
        if "SEED" in nested:
            cfg.SEED = int(nested["SEED"])
    return cfg


def default_hard_list():
    candidates = sorted(
        ROOT_DIR.glob("train_log/**/common15_trajectory_summary_for_gpt.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def main():
    args = parse_args()
    models = get_models(args)
    hard_list_value = args.hard_list or ""
    hard_filenames = []
    hard_list_path = None
    if hard_list_value:
        hard_list_path, hard_filenames = load_hard_case_filenames(hard_list_value)
    else:
        auto_hard_list = default_hard_list()
        if auto_hard_list is not None:
            hard_list_path, hard_filenames = load_hard_case_filenames(str(auto_hard_list))
        elif args.hard_only:
            raise FileNotFoundError(
                "No default hard-case list was found under train_log/. "
                "Provide one with --hard-list."
            )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else ROOT_DIR / "train_log" / f"ppo_rl_fixed_protocol_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    thresholds = {
        "endpoint": args.endpoint_threshold,
        "safety": args.safety_threshold,
        "collision": args.collision_threshold,
    }

    print("=" * 80)
    print("PPO-RL fixed protocol evaluation")
    print(f"Output: {output_dir}")
    print(f"Hard-case list: {hard_list_path if hard_list_path is not None else 'not provided; full split only'}")
    for item in models:
        print(f"- {item['tag']}: {item['checkpoint']}")
    print("=" * 80)

    if args.dry_run:
        print("[Dry run] No policy loaded and no evaluation executed.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_dataset(device)
    hard_indices = dataset_indices_by_filename(dataset, hard_filenames) if hard_filenames else []
    full_indices = list(range(len(dataset)))
    if args.hard_only:
        splits = [("hard15", hard_indices)]
    elif hard_filenames:
        splits = [("full", full_indices), ("hard15", hard_indices)]
    else:
        splits = [("full", full_indices)]

    if hard_filenames:
        write_csv(output_dir / "hard_case_list_used.csv", [{"filename": name} for name in hard_filenames], ["filename"])

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "dataset_size": len(dataset),
        "hard_case_count": len(hard_indices),
        "hard_case_list": str(hard_list_path) if hard_list_path is not None else "",
        "thresholds": thresholds,
        "hard_only": args.hard_only,
        "models": [],
    }

    protocol_summaries = []
    full_rows_by_model = {}
    hard_rows_by_model = {}

    for model_info in models:
        run_config, run_dir = load_run_config(model_info["checkpoint"])
        cfg = cfg_for_model(run_config)
        manifest["models"].append(
            {
                "tag": model_info["tag"],
                "checkpoint": str(model_info["checkpoint"]),
                "run_dir": str(run_dir),
                "run_config_found": bool(run_config),
                "effective_goal_mode": "normal" if cfg.ENABLE_GOAL_CONDITION else "disabled",
                "effective_attention_mode": "normal" if cfg.ENABLE_ATTENTION else "disabled",
                "seed": int(cfg.SEED),
            }
        )

        print("\n" + "-" * 80)
        print(f"Loading {model_info['tag']}")
        print(f"Checkpoint: {model_info['checkpoint']}")
        print(
            f"Config override -> goal={cfg.ENABLE_GOAL_CONDITION}, "
            f"attention={cfg.ENABLE_ATTENTION}, seed={cfg.SEED}"
        )
        model = PPO.load(str(model_info["checkpoint"]), device=device)

        for split_name, indices in splits:
            rows = evaluate_indices(model, dataset, cfg, indices, model_info["tag"], split_name)
            write_split_outputs(output_dir, model_info["tag"], split_name, rows)

            summary = summarize_rows(model_info, rows)
            annotated = annotate_summary(summary, split_name, thresholds)
            protocol_summaries.append(annotated)
            if split_name == "full":
                full_rows_by_model[model_info["tag"]] = rows
            else:
                hard_rows_by_model[model_info["tag"]] = rows

            print(
                f"{model_info['tag']}/{split_name}: "
                f"endpoint={float(annotated['endpoint_success_rate']):.3f}, "
                f"safety={float(annotated['safety_success_rate']):.3f}, "
                f"collision={float(annotated['collision_rate']):.3f}, "
                f"collisions={annotated['collision_count']}, "
                f"pass={annotated['protocol_pass']}"
            )

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8-sig",
    )
    write_csv(output_dir / "protocol_summary.csv", protocol_summaries, PROTOCOL_SUMMARY_FIELDS)
    if full_rows_by_model:
        write_overlap(output_dir / "full_overlap", full_rows_by_model)
    if hard_rows_by_model:
        write_overlap(output_dir / "hard15_overlap", hard_rows_by_model)
    write_protocol_report(output_dir, protocol_summaries, hard_filenames, thresholds)

    print("\n" + "=" * 80)
    print(f"Saved PPO-RL fixed protocol evaluation to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
