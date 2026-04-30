import argparse
import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from stable_baselines3 import PPO

from configs.config import Config
from train_v10b_risk_classifier import V10BRiskClassifier, evaluate_classifier_protocol
from v10b_distill_common import (
    BASE_MODEL_TAG,
    build_v10b_shield_params,
    evaluate_model_suites,
    get_model_info_by_tag,
    load_eval_dataset,
    load_hard_case_info,
    save_json,
    seed_everything,
)


ROOT_DIR = Path(__file__).resolve().parent
EPOCH_PATTERN = re.compile(r"epoch_(\d+)\.pt$", re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate v10b risk classifier thresholds without retraining.")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--base-model-tag", default=BASE_MODEL_TAG)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--tau-intervene-list", default="0.05,0.10,0.15,0.20,0.30,0.40,0.50")
    parser.add_argument("--tau-critical-list", default="0.05,0.10,0.20,0.30")
    parser.add_argument(
        "--raw-critical-override-mode",
        default="all",
        choices=["all", "overlap_or_ttc", "overlap_only", "none"],
        help="How the raw v10b critical rule participates in override gating.",
    )
    parser.add_argument(
        "--selection-profile",
        default="strict_target",
        choices=["strict_target", "practical_gate"],
        help="How to choose the best row from calibration results.",
    )
    parser.add_argument("--eval-full-limit", type=int, default=0)
    parser.add_argument("--eval-hard-limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def default_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT_DIR / "train_log" / f"v10b_risk_calib_{timestamp}"


def parse_float_list(raw: str) -> List[float]:
    return [float(item.strip()) for item in str(raw).split(",") if item.strip()]


def checkpoint_epoch_from_name(path: Path) -> int:
    match = EPOCH_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"Unsupported checkpoint filename: {path.name}")
    return int(match.group(1))


def list_epoch_checkpoints(checkpoint_dir: Path) -> List[Tuple[int, Path]]:
    pairs = []
    for path in checkpoint_dir.glob("epoch_*.pt"):
        pairs.append((checkpoint_epoch_from_name(path), path.resolve()))
    pairs.sort(key=lambda item: item[0])
    if not pairs:
        raise FileNotFoundError(f"No epoch_*.pt checkpoints found in {checkpoint_dir}")
    return pairs


def passes_core_protocol(row: Dict[str, float]) -> bool:
    return (
        float(row["full217_collision_rate"]) <= 1e-12
        and float(row["hard15_collision_rate"]) <= 1e-12
        and float(row["full217_merge_success_rate"]) >= 0.86
        and float(row["full217_endpoint_success_rate"]) >= 0.91
        and float(row["full217_safety_success_rate"]) >= 0.91
    )


def passes_thresholds(row: Dict[str, float]) -> bool:
    return (
        passes_core_protocol(row)
        and float(row["full217_shield_rate"]) < 0.3332
    )


def best_key(row: Dict[str, float], selection_profile: str) -> Tuple[float, float, float, float, float]:
    if selection_profile == "strict_target":
        if passes_thresholds(row):
            return (
                1.0,
                -float(row["full217_shield_rate"]),
                -float(row["full217_false_negative_rate"]),
                -float(row["hard15_critical_false_negative_rate"]),
                -float(row["checkpoint_epoch"]),
            )
        return (
            0.0,
            -float(row["full217_collision_rate"]),
            -float(row["hard15_collision_rate"]),
            -float(row["full217_false_negative_rate"]),
            -float(row["full217_shield_rate"]),
        )

    if selection_profile == "practical_gate":
        if passes_core_protocol(row):
            return (
                1.0,
                -float(row["full217_shield_rate"]),
                -float(row["full217_false_negative_rate"]),
                -float(row["full217_critical_false_negative_rate"]),
                -float(row["checkpoint_epoch"]),
            )
        return (
            0.0,
            -float(row["full217_collision_rate"]),
            -float(row["hard15_collision_rate"]),
            -float(row["full217_merge_success_rate"]),
            -float(row["full217_shield_rate"]),
        )

    raise ValueError(f"Unsupported selection_profile: {selection_profile}")


def load_classifier_checkpoint(checkpoint_path: Path, device: torch.device) -> Tuple[V10BRiskClassifier, int]:
    payload = torch.load(checkpoint_path, map_location=device)
    model = V10BRiskClassifier(input_dim=24).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    epoch = int(payload.get("epoch", checkpoint_epoch_from_name(checkpoint_path)))
    return model, epoch


def write_csv(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_baseline(base_model, dataset, cfg: Config, full_indices: Sequence[int], hard_indices: Sequence[int], shield_params: Dict) -> Dict:
    suites = evaluate_model_suites(
        model=base_model,
        dataset=dataset,
        cfg=cfg,
        model_tag=BASE_MODEL_TAG,
        full_indices=full_indices,
        hard_indices=hard_indices,
        shield_params=shield_params,
    )
    return {
        "full217": suites["full"]["shield_summary"],
        "hard15": suites["hard15"]["shield_summary"],
    }


def main():
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else default_output_dir().resolve()
    tau_intervene_list = parse_float_list(args.tau_intervene_list)
    tau_critical_list = parse_float_list(args.tau_critical_list)
    checkpoints = list_epoch_checkpoints(checkpoint_dir)

    if args.dry_run:
        print("=" * 80)
        print("v10b classifier calibration dry-run")
        print(f"checkpoint_dir={checkpoint_dir}")
        print(f"output_dir={output_dir}")
        print(f"checkpoints={[path.name for _, path in checkpoints]}")
        print(f"tau_intervene_list={tau_intervene_list}")
        print(f"tau_critical_list={tau_critical_list}")
        print(f"raw_critical_override_mode={args.raw_critical_override_mode}")
        print(f"selection_profile={args.selection_profile}")
        print("=" * 80)
        return

    seed_everything(args.seed, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()
    shield_params = build_v10b_shield_params()
    dataset = load_eval_dataset(device=device)
    _, _, hard_indices, hard_index_set = load_hard_case_info(dataset)
    hard_indices = list(hard_indices)
    full_indices = list(range(len(dataset)))
    if args.eval_hard_limit:
        hard_indices = hard_indices[: int(args.eval_hard_limit)]
    if args.eval_full_limit:
        full_indices = full_indices[: int(args.eval_full_limit)]

    output_dir.mkdir(parents=True, exist_ok=True)
    calibration_rows: List[Dict] = []
    best_row = None
    best_score = None

    base_model = PPO.load(str(get_model_info_by_tag(args.base_model_tag)["checkpoint"]), device=device)
    baseline_summary = summarize_baseline(base_model, dataset, cfg, full_indices, hard_indices, shield_params)
    save_json(output_dir / "baseline_summary.json", baseline_summary)

    for checkpoint_epoch, checkpoint_path in checkpoints:
        classifier, loaded_epoch = load_classifier_checkpoint(checkpoint_path, device)
        for tau_intervene in tau_intervene_list:
            for tau_critical in tau_critical_list:
                hard_metrics = evaluate_classifier_protocol(
                    base_model=base_model,
                    classifier=classifier,
                    dataset=dataset,
                    cfg=cfg,
                    traj_indices=hard_indices,
                    shield_params=shield_params,
                    tau_intervene=float(tau_intervene),
                    tau_critical=float(tau_critical),
                    device=device,
                    critical_overrides_gate=True,
                    use_raw_critical_rule=True,
                    raw_critical_override_mode=args.raw_critical_override_mode,
                )
                full_metrics = evaluate_classifier_protocol(
                    base_model=base_model,
                    classifier=classifier,
                    dataset=dataset,
                    cfg=cfg,
                    traj_indices=full_indices,
                    shield_params=shield_params,
                    tau_intervene=float(tau_intervene),
                    tau_critical=float(tau_critical),
                    device=device,
                    critical_overrides_gate=True,
                    use_raw_critical_rule=True,
                    raw_critical_override_mode=args.raw_critical_override_mode,
                )
                row = {
                    "checkpoint_epoch": int(loaded_epoch),
                    "checkpoint_path": str(checkpoint_path),
                    "raw_critical_override_mode": args.raw_critical_override_mode,
                    "tau_intervene": float(tau_intervene),
                    "tau_critical": float(tau_critical),
                    "full217_merge_success_rate": full_metrics["merge_success_rate"],
                    "full217_endpoint_success_rate": full_metrics["endpoint_success_rate"],
                    "full217_safety_success_rate": full_metrics["safety_success_rate"],
                    "full217_collision_rate": full_metrics["collision_rate"],
                    "full217_collision_count": full_metrics["collision_count"],
                    "full217_shield_rate": full_metrics["predicted_intervention_rate"],
                    "full217_false_negative_rate": full_metrics["false_negative_rate"],
                    "full217_critical_false_negative_rate": full_metrics["critical_false_negative_rate"],
                    "full217_raw_critical_override_rate": full_metrics["raw_critical_override_rate"],
                    "hard15_merge_success_rate": hard_metrics["merge_success_rate"],
                    "hard15_endpoint_success_rate": hard_metrics["endpoint_success_rate"],
                    "hard15_safety_success_rate": hard_metrics["safety_success_rate"],
                    "hard15_collision_rate": hard_metrics["collision_rate"],
                    "hard15_collision_count": hard_metrics["collision_count"],
                    "hard15_shield_rate": hard_metrics["predicted_intervention_rate"],
                    "hard15_false_negative_rate": hard_metrics["false_negative_rate"],
                    "hard15_critical_false_negative_rate": hard_metrics["critical_false_negative_rate"],
                    "hard15_raw_critical_override_rate": hard_metrics["raw_critical_override_rate"],
                }
                row["passes_core_protocol"] = passes_core_protocol(row)
                row["passes_thresholds"] = passes_thresholds(row)
                calibration_rows.append(row)
                current_score = best_key(row, args.selection_profile)
                if best_score is None or current_score > best_score:
                    best_score = current_score
                    best_row = row

    write_csv(output_dir / "calibration_summary.csv", calibration_rows)
    save_json(
        output_dir / "best_config.json",
        {
            "best_row": best_row,
            "best_score": list(best_score) if best_score is not None else [],
            "passes_thresholds": bool(best_row["passes_thresholds"]) if best_row else False,
            "baseline_summary_path": str((output_dir / "baseline_summary.json").resolve()),
        },
    )
    save_json(
        output_dir / "run_summary.json",
        {
            "checkpoint_dir": str(checkpoint_dir),
            "checkpoint_count": len(checkpoints),
            "row_count": len(calibration_rows),
            "tau_intervene_list": tau_intervene_list,
            "tau_critical_list": tau_critical_list,
            "raw_critical_override_mode": args.raw_critical_override_mode,
            "selection_profile": args.selection_profile,
            "passes_core_protocol": bool(best_row["passes_core_protocol"]) if best_row else False,
            "passes_thresholds": bool(best_row["passes_thresholds"]) if best_row else False,
        },
    )
    print(f"[*] Calibration summary: {output_dir / 'calibration_summary.csv'}")
    print(f"[*] Best config: {output_dir / 'best_config.json'}")


if __name__ == "__main__":
    main()
