import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from tqdm import tqdm

from configs.config import Config
from envs.merging_env import MergingEnv
from evaluation.failure_case_full_evaluate import SingleTrajDataset
from evaluation.safety_shield_evaluate import (
    action_risk,
    evaluate_single_trajectory_with_shield,  # noqa: F401
    is_merge_phase,
    shield_action,
    target_lane_vehicle_metrics,
)
from model.attention_net import AttentionFeaturesExtractor, GoalConditionedMLPFeaturesExtractor  # noqa: F401
from v10b_distill_common import (
    BASE_MODEL_TAG,
    V10B_SHIELD_VARIANT,
    build_v10b_shield_params,
    get_model_info_by_tag,
    load_eval_dataset,
    load_hard_case_info,
    save_json,
    seed_everything,
)


ROOT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export per-step v10b teacher samples without changing training or existing evaluation scripts."
    )
    parser.add_argument("--base-model-tag", default=BASE_MODEL_TAG)
    parser.add_argument("--hard-list", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--normal-safe-keep-prob", type=float, default=0.15)
    parser.add_argument("--limit-trajectories", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def default_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT_DIR / "train_log" / f"v10b_teacher_dataset_{timestamp}"


def sample_kind(intervened: bool, warning: bool) -> str:
    if intervened:
        return "intervened"
    if warning:
        return "warning"
    return "normal_safe"


def risk_side_label(risk_vehicle: str) -> str:
    if risk_vehicle in {"L6_lead", "L5_lead"}:
        return "lead"
    if risk_vehicle in {"L5_follow", "L6_follow"}:
        return "follow"
    return "none"


def is_v10b_policy_veto(policy_action, risk_vehicle: str, lead_gap: float, lead_thw: float, shield_params: dict) -> bool:
    return (
        float(policy_action[0]) > shield_params.get("merge_recovery_policy_veto_x", 0.25)
        and risk_vehicle == "L5_follow"
        and (
            float(lead_gap) < shield_params.get("merge_recovery_policy_veto_lead_gap", 100.0)
            or float(lead_thw) < shield_params.get("merge_recovery_policy_veto_lead_thw", 5.0)
        )
    )


def risk_reason_label(
    original_score: dict,
    shield_info: dict,
    *,
    merge_recovery_active: bool,
    veto_active: bool,
    shield_params: dict,
) -> str:
    if veto_active:
        return "veto"
    if merge_recovery_active:
        return "recovery"
    if bool(original_score.get("predicted_overlap", False)):
        return "overlap"
    if float(original_score.get("min_ttc", 20.0)) < shield_params.get("lead_ttc_min", 3.0):
        return "low_ttc"
    min_thw = float(original_score.get("min_thw", 10.0))
    if min_thw < max(shield_params.get("lead_thw_min", 1.0), shield_params.get("follow_thw_min", 0.8)):
        return "low_thw"
    return "none"


def keep_sample(kind: str, rng: np.random.Generator, keep_prob: float) -> bool:
    if kind in {"intervened", "warning"}:
        return True
    return bool(rng.random() < keep_prob)


def write_teacher_summary(path: Path, counters: dict) -> None:
    rows = []
    for split_name in sorted(counters.keys()):
        for kind in ("intervened", "warning", "normal_safe"):
            raw_count = int(counters[split_name][kind]["raw"])
            kept_count = int(counters[split_name][kind]["kept"])
            keep_ratio = (kept_count / raw_count) if raw_count else 0.0
            rows.append(
                {
                    "split": split_name,
                    "sample_kind": kind,
                    "raw_count": raw_count,
                    "kept_count": kept_count,
                    "keep_ratio": keep_ratio,
                }
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "sample_kind", "raw_count", "kept_count", "keep_ratio"],
        )
        writer.writeheader()
        writer.writerows(rows)


def export_teacher_dataset(args) -> Path:
    output_dir = Path(args.output_dir).resolve() if args.output_dir else default_output_dir().resolve()
    model_info = get_model_info_by_tag(args.base_model_tag)
    shield_params = build_v10b_shield_params()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()

    seed_everything(args.seed, deterministic=True)
    rng = np.random.default_rng(args.seed)

    dataset = load_eval_dataset(device=device)
    hard_list_path, hard_filenames, hard_indices, hard_index_set = load_hard_case_info(
        dataset,
        hard_list_path=args.hard_list or None,
    )
    all_indices = list(range(len(dataset)))
    if args.limit_trajectories and args.limit_trajectories > 0:
        all_indices = all_indices[: int(args.limit_trajectories)]

    print("=" * 80)
    print("v10b teacher dataset export")
    print(f"Base model: {args.base_model_tag}")
    print(f"Checkpoint: {model_info['checkpoint']}")
    print(f"Shield variant: {V10B_SHIELD_VARIANT}")
    print(f"Output dir: {output_dir}")
    print(f"Dataset trajectories: {len(dataset)}")
    print(f"Selected trajectories: {len(all_indices)}")
    print(f"Hard-15 file: {hard_list_path}")
    print(f"Hard-15 matches: {len(hard_indices)}")
    print(f"Normal-safe keep prob: {args.normal_safe_keep_prob}")
    print("=" * 80)
    if args.dry_run:
        print("[Dry run] No model loaded and no samples exported.")
        return output_dir / "teacher_dataset.pt"

    model = PPO.load(str(model_info["checkpoint"]), device=device)
    samples = []
    counters = defaultdict(lambda: defaultdict(lambda: {"raw": 0, "kept": 0}))

    for traj_index in tqdm(all_indices, desc="export_teacher"):
        traj = dataset[traj_index]
        single_dataset = SingleTrajDataset(traj, dataset.expert_mean, dataset.expert_std)
        env = MergingEnv(single_dataset)
        env.collision_margin = 1.0

        obs, _ = env.reset(seed=cfg.SEED + traj_index)
        terminated = False
        truncated = False
        max_steps = len(env.current_traj["ego_pos"]) + 50
        step_count = 0
        shield_interventions = 0
        shield_warnings = 0
        consecutive_interventions = 0
        safe_steps = 0
        split_name = "hard15" if traj_index in hard_index_set else "full"
        filename = traj.get("filename", f"trajectory_{traj_index}.csv")

        while not (terminated or truncated) and step_count < max_steps:
            policy_action, _ = model.predict(obs, deterministic=True)
            policy_action = np.asarray(policy_action, dtype=np.float32).reshape(-1)
            original_score = action_risk(env, policy_action, policy_action, shield_params)
            target_metrics = target_lane_vehicle_metrics(env)
            merge_phase = is_merge_phase(env, policy_action)
            shielded_action, shield_info = shield_action(
                env=env,
                policy_action=policy_action,
                params=shield_params,
                step_count=step_count,
                shield_interventions=shield_interventions,
                consecutive_interventions=consecutive_interventions,
                safe_steps=safe_steps,
            )
            shielded_action = np.asarray(shielded_action, dtype=np.float32).reshape(-1)
            selected_score = action_risk(env, shielded_action, policy_action, shield_params)

            if shield_info["intervened"]:
                shield_interventions += 1
                consecutive_interventions += 1
            else:
                consecutive_interventions = 0
            if shield_info.get("warning", False):
                shield_warnings += 1

            next_obs, _, terminated, truncated, info = env.step(shielded_action)
            collided_now = bool(info.get("is_collided", False) or getattr(env, "has_collided_this_episode", False))
            if (
                not collided_now
                and float(info.get("eval_min_ttc", 20.0)) >= shield_params.get("lead_ttc_min", 3.0)
                and float(info.get("eval_min_thw", 10.0)) >= shield_params.get("follow_thw_min", 0.8)
            ):
                safe_steps += 1
            else:
                safe_steps = 0

            kind = sample_kind(bool(shield_info["intervened"]), bool(shield_info.get("warning", False)))
            counters[split_name][kind]["raw"] += 1
            keep = keep_sample(kind, rng, args.normal_safe_keep_prob)

            if keep:
                counters[split_name][kind]["kept"] += 1
                lead_metrics = target_metrics["lead"]
                follow_metrics = target_metrics["follow"]
                risk_vehicle = original_score.get("critical_vehicle") or original_score.get("nearest", "")
                merge_recovery_active = bool(shield_info.get("candidate") == "merge_recovery_lateral")
                veto_active = is_v10b_policy_veto(
                    policy_action=policy_action,
                    risk_vehicle=risk_vehicle,
                    lead_gap=float(lead_metrics["gap"]),
                    lead_thw=float(lead_metrics["thw"]),
                    shield_params=shield_params,
                )
                reason_label = risk_reason_label(
                    original_score,
                    shield_info,
                    merge_recovery_active=merge_recovery_active,
                    veto_active=veto_active,
                    shield_params=shield_params,
                )
                sample = {
                    "obs": np.asarray(obs, dtype=np.float32).copy(),
                    "policy_action": policy_action.astype(np.float32).copy(),
                    "shield_action": shielded_action.astype(np.float32).copy(),
                    "executed_action": shielded_action.astype(np.float32).copy(),
                    "shield_intervened": bool(shield_info["intervened"]),
                    "shield_warning": bool(shield_info.get("warning", False)),
                    "risk_vehicle": risk_vehicle,
                    "risk_reason": str(shield_info.get("reason", "")),
                    "original_risk": float(original_score["risk"]),
                    "selected_risk": float(selected_score["risk"]),
                    "lead_gap": float(lead_metrics["gap"]),
                    "lead_thw": float(lead_metrics["thw"]),
                    "follow_gap": float(follow_metrics["gap"]),
                    "follow_thw": float(follow_metrics["thw"]),
                    "merge_phase": bool(merge_phase),
                    "merge_recovery_active": merge_recovery_active,
                    "split": split_name,
                    "traj_index": int(traj_index),
                    "filename": filename,
                    "step": int(step_count),
                    "collision_flag": bool(collided_now),
                    "endpoint_flag": bool(info.get("is_endpoint_success", False)),
                    "merge_flag": bool(info.get("is_merge_success", False)),
                    "sample_kind": kind,
                    "critical_risk": bool(original_score.get("critical", False)),
                    "predicted_overlap": bool(original_score.get("predicted_overlap", False)),
                    "risk_side_label": risk_side_label(risk_vehicle),
                    "risk_reason_label": reason_label,
                    "delta_action": (shielded_action - policy_action).astype(np.float32).copy(),
                    "normal_safe_mask": bool(kind == "normal_safe"),
                }
                samples.append(sample)

            obs = next_obs
            step_count += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "teacher_dataset.pt"
    summary_path = output_dir / "teacher_summary.csv"
    manifest_path = output_dir / "teacher_manifest.json"

    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "base_model_tag": args.base_model_tag,
        "base_checkpoint": str(model_info["checkpoint"]),
        "base_note": model_info.get("note", ""),
        "shield_variant": V10B_SHIELD_VARIANT,
        "shield_params": shield_params,
        "seed": int(args.seed),
        "device": str(device),
        "normal_safe_keep_prob": float(args.normal_safe_keep_prob),
        "hard_list_path": str(hard_list_path),
        "hard_filenames": hard_filenames,
        "hard_indices": [int(idx) for idx in hard_indices],
        "selected_trajectories": [int(idx) for idx in all_indices],
        "total_trajectories": len(dataset),
        "selected_trajectory_count": len(all_indices),
        "retained_sample_count": len(samples),
        "counters": {
            split_name: {
                kind: {
                    "raw": int(kind_counts["raw"]),
                    "kept": int(kind_counts["kept"]),
                }
                for kind, kind_counts in split_counts.items()
            }
            for split_name, split_counts in counters.items()
        },
    }

    with dataset_path.open("wb") as f:
        torch.save({"meta": meta, "samples": samples}, f)
    save_json(manifest_path, meta)
    write_teacher_summary(summary_path, counters)

    print(f"[*] Exported teacher dataset: {dataset_path}")
    print(f"[*] Manifest: {manifest_path}")
    print(f"[*] Summary: {summary_path}")
    print(f"[*] Retained samples: {len(samples)}")
    return dataset_path


def main():
    args = parse_args()
    export_teacher_dataset(args)


if __name__ == "__main__":
    main()
