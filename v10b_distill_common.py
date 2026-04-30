import csv
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from stable_baselines3 import PPO


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_HARD_CASE_LIST = (
    ROOT_DIR / "train_log" / "failure_case_full_eval_20260427_205244" / "common15_trajectory_summary_for_gpt.csv"
)
BASE_MODEL_TAG = "U220_D230_epoch290"
V10B_SHIELD_VARIANT = "v10b_leadgap_policy_veto_recovery"
DEFAULT_THRESHOLDS = {"endpoint": 0.95, "safety": 0.90, "collision": 0.01}


def seed_everything(seed: int, deterministic: bool = True) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_csv_row(path: Path, row: Dict, fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fieldnames or row.keys())
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def get_model_info_by_tag(tag: str = BASE_MODEL_TAG) -> Dict:
    from evaluation.failure_case_full_evaluate import TARGET_MODELS

    for item in TARGET_MODELS:
        if item["tag"] == tag:
            return {
                "tag": item["tag"],
                "checkpoint": Path(item["checkpoint"]).resolve(),
                "note": item.get("note", ""),
            }
    raise ValueError(f"Unknown model tag: {tag}")


def build_v10b_shield_params() -> Dict:
    return {
        "shield_variant": V10B_SHIELD_VARIANT,
        "prediction_horizon": 1.0,
        "lead_ttc_min": 3.0,
        "follow_ttc_min": 3.0,
        "lead_thw_min": 1.0,
        "follow_thw_min": 0.8,
        "risk_improvement_margin": 0.15,
        "max_intervention_rate": 0.60,
        "max_consecutive_interventions": 10,
        "critical_ttc_min": 1.0,
        "critical_thw_min": 0.25,
        "critical_follow_thw_min": 0.25,
        "recovery_blend_enabled": True,
        "recovery_safe_steps": 5,
        "recovery_policy_steps": 8,
        "follow_burst_thw_trigger": 0.5,
        "follow_burst_ttc_trigger": 2.0,
        "follow_burst_gap_min": 0.0,
        "follow_burst_action_y": 1.0,
        "follow_burst_lead_gap_min": 15.0,
        "follow_burst_lead_ttc_min": 1.0,
        "follow_burst_lead_thw_min": 0.30,
        "merge_recovery_policy_x_trigger": -0.03,
        "merge_recovery_min_progress": 0.15,
        "merge_recovery_target_x": -0.20,
        "merge_recovery_x_options": (-1.0, -0.75, -0.50, -0.35, -0.25, -0.15, -0.10, -0.05),
        "merge_recovery_risk_slack": 1.50,
        "merge_recovery_aux_slack": 2.0,
        "merge_recovery_policy_veto_x": 0.25,
        "merge_recovery_policy_veto_lead_gap": 100.0,
        "merge_recovery_policy_veto_lead_thw": 5.0,
    }


def load_eval_dataset(device: torch.device):
    from evaluation.failure_case_full_evaluate import load_dataset

    return load_dataset(device=device)


def load_hard_case_info(dataset, hard_list_path: Optional[str] = None) -> Tuple[Path, List[str], List[int], set]:
    from evaluation.hard_case_protocol_evaluate import dataset_indices_by_filename, load_hard_case_filenames

    hard_list_path = hard_list_path or str(DEFAULT_HARD_CASE_LIST)
    resolved_path, hard_filenames = load_hard_case_filenames(hard_list_path)
    hard_indices = dataset_indices_by_filename(dataset, hard_filenames)
    return resolved_path, hard_filenames, hard_indices, set(hard_indices)


def limit_indices(indices: Sequence[int], limit: Optional[int]) -> List[int]:
    limited = list(indices)
    if limit is not None:
        limited = limited[: int(limit)]
    return limited


def evaluate_model_suites(
    model: PPO,
    dataset,
    cfg,
    model_tag: str,
    full_indices: Sequence[int],
    hard_indices: Sequence[int],
    shield_params: Dict,
    thresholds: Optional[Dict] = None,
) -> Dict[str, Dict]:
    from evaluation.failure_case_full_evaluate import evaluate_single_trajectory, summarize_rows
    from evaluation.hard_case_protocol_evaluate import annotate_summary
    from evaluation.safety_shield_evaluate import evaluate_single_trajectory_with_shield, summarize_with_shield

    thresholds = dict(DEFAULT_THRESHOLDS, **(thresholds or {}))
    model_info = get_model_info_by_tag(model_tag)

    def evaluate_split(indices: Sequence[int], split_name: str) -> Dict[str, Dict]:
        no_shield_rows = [
            {**evaluate_single_trajectory(model, dataset, cfg, traj_index), "model_tag": model_tag}
            for traj_index in indices
        ]
        shield_rows = [
            {**evaluate_single_trajectory_with_shield(model, dataset, cfg, traj_index, shield_params), "model_tag": model_tag}
            for traj_index in indices
        ]
        no_shield_summary = annotate_summary(summarize_rows(model_info, no_shield_rows), split_name, thresholds)
        shield_summary = summarize_with_shield(model_info, shield_rows, split_name, thresholds)
        return {
            "no_shield_rows": no_shield_rows,
            "shield_rows": shield_rows,
            "no_shield_summary": no_shield_summary,
            "shield_summary": shield_summary,
        }

    return {
        "hard15": evaluate_split(hard_indices, "hard15"),
        "full": evaluate_split(full_indices, "full"),
    }


def flatten_eval_summaries(eval_summaries: Dict[str, Dict]) -> Dict[str, float]:
    row: Dict[str, float] = {}
    for split_name, split_payload in eval_summaries.items():
        for variant_name in ("no_shield_summary", "shield_summary"):
            prefix = f"{split_name}_{'shield' if variant_name == 'shield_summary' else 'no_shield'}"
            summary = split_payload[variant_name]
            for key in (
                "merge_success_rate",
                "endpoint_success_rate",
                "safety_success_rate",
                "collision_rate",
                "collision_count",
            ):
                row[f"{prefix}_{key}"] = summary.get(key, "")
            if variant_name == "shield_summary":
                row[f"{prefix}_shield_intervention_rate"] = summary.get("shield_intervention_rate", "")
    return row


def build_best_key(eval_summaries: Dict[str, Dict]) -> Tuple[float, float, float, float, float]:
    full_shield = eval_summaries["full"]["shield_summary"]
    full_no_shield = eval_summaries["full"]["no_shield_summary"]
    shield_collision = float(full_shield.get("collision_rate", 1.0))
    shield_intervention_rate = float(full_shield.get("shield_intervention_rate", 1.0))
    no_shield_collision = float(full_no_shield.get("collision_rate", 1.0))
    no_shield_endpoint = float(full_no_shield.get("endpoint_success_rate", 0.0))
    no_shield_merge = float(full_no_shield.get("merge_success_rate", 0.0))
    zero_collision_pass = 1.0 if shield_collision <= 1e-12 else 0.0
    return (
        zero_collision_pass,
        -shield_intervention_rate,
        -no_shield_collision,
        no_shield_endpoint,
        no_shield_merge,
    )


def load_teacher_payload(path: Path) -> Dict:
    payload = torch.load(path, map_location="cpu")
    if "samples" not in payload or "meta" not in payload:
        raise ValueError(f"Malformed teacher dataset payload: {path}")
    return payload


def resolve_base_checkpoint_from_teacher(payload: Dict) -> Path:
    checkpoint = payload["meta"].get("base_checkpoint", "")
    if not checkpoint:
        raise ValueError("Teacher dataset payload missing base_checkpoint")
    return Path(checkpoint).resolve()

