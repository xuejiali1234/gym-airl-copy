import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from tqdm import tqdm

curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(curr_dir)
sys.path.append(root_dir)

from configs.config import Config
from evaluation.performance_evaluate_transfer import (
    build_default_model_path,
    build_default_stats_paths,
    load_eval_dataset_with_fixed_stats,
    load_stats_dataset,
)
from evaluation.safety_shield_evaluate import evaluate_single_trajectory_with_shield


def build_shield_params(variant, prediction_horizon, lead_ttc_min, follow_ttc_min, lead_thw_min, follow_thw_min):
    risk_improvement_margin = 0.15 if variant in ("v3_gate_margin", "v4_combo") else 0.0
    if variant == "v6c_margin020":
        risk_improvement_margin = 0.20
    max_intervention_rate = 1.0
    if variant == "v4_combo":
        max_intervention_rate = 0.50
    recovery_enabled = variant in (
        "v5_critical_override",
        "v6b_recovery_merge",
        "v6c_margin020",
        "v7a_follow_low_thw",
        "v7b_merge_preserving_warning",
        "v8a_follow_burst",
        "v9a_merge_recovery",
        "v9b_near_aux_recovery",
        "v10a_policy_veto_recovery",
        "v10b_leadgap_policy_veto_recovery",
    )
    recovery_safe_steps = 3 if variant in ("v6b_recovery_merge", "v6c_margin020") else 5
    recovery_policy_steps = 6 if variant in ("v6b_recovery_merge", "v6c_margin020") else 8

    shield_params = {
        "shield_variant": variant,
        "prediction_horizon": prediction_horizon,
        "lead_ttc_min": lead_ttc_min,
        "follow_ttc_min": follow_ttc_min,
        "lead_thw_min": lead_thw_min,
        "follow_thw_min": follow_thw_min,
        "risk_improvement_margin": risk_improvement_margin,
        "max_intervention_rate": max_intervention_rate,
        "max_consecutive_interventions": 10
        if variant
        in (
            "v4_combo",
            "v5_critical_override",
            "v6a_follow_emergency",
            "v6b_recovery_merge",
            "v6c_margin020",
            "v7a_follow_low_thw",
            "v7b_merge_preserving_warning",
            "v8a_follow_burst",
            "v9a_merge_recovery",
            "v9b_near_aux_recovery",
            "v10a_policy_veto_recovery",
            "v10b_leadgap_policy_veto_recovery",
        )
        else 10**9,
        "critical_ttc_min": 1.0,
        "critical_thw_min": 0.25,
        "critical_follow_thw_min": 0.25,
        "recovery_blend_enabled": recovery_enabled,
        "recovery_safe_steps": recovery_safe_steps,
        "recovery_policy_steps": recovery_policy_steps,
        "follow_burst_thw_trigger": 0.5,
        "follow_burst_ttc_trigger": 2.0,
        "follow_burst_gap_min": 0.0,
        "follow_burst_action_y": 1.0,
        "follow_burst_lead_gap_min": 15.0,
        "follow_burst_lead_ttc_min": 1.0,
        "follow_burst_lead_thw_min": 0.30,
        "merge_recovery_policy_x_trigger": -0.03,
        "merge_recovery_min_progress": 0.35,
        "merge_recovery_target_x": -0.10,
        "merge_recovery_x_options": (-0.15, -0.10, -0.05),
        "merge_recovery_risk_slack": 1.05,
        "merge_recovery_aux_slack": 0.0,
        "merge_recovery_policy_veto_x": 0.25,
    }

    if variant == "v9b_near_aux_recovery":
        shield_params.update(
            {
                "merge_recovery_aux_slack": 2.0,
                "merge_recovery_min_progress": 0.15,
                "merge_recovery_target_x": -0.20,
                "merge_recovery_x_options": (-1.0, -0.75, -0.50, -0.35, -0.25, -0.15, -0.10, -0.05),
                "merge_recovery_risk_slack": 1.50,
            }
        )
    if variant == "v10a_policy_veto_recovery":
        shield_params.update(
            {
                "merge_recovery_aux_slack": 2.0,
                "merge_recovery_min_progress": 0.15,
                "merge_recovery_target_x": -0.20,
                "merge_recovery_x_options": (-1.0, -0.75, -0.50, -0.35, -0.25, -0.15, -0.10, -0.05),
                "merge_recovery_risk_slack": 1.50,
                "merge_recovery_policy_veto_x": 0.25,
            }
        )
    if variant == "v10b_leadgap_policy_veto_recovery":
        shield_params.update(
            {
                "merge_recovery_aux_slack": 2.0,
                "merge_recovery_min_progress": 0.15,
                "merge_recovery_target_x": -0.20,
                "merge_recovery_x_options": (-1.0, -0.75, -0.50, -0.35, -0.25, -0.15, -0.10, -0.05),
                "merge_recovery_risk_slack": 1.50,
                "merge_recovery_policy_veto_x": 0.25,
                "merge_recovery_policy_veto_lead_gap": 100.0,
                "merge_recovery_policy_veto_lead_thw": 5.0,
            }
        )
    return shield_params


def summarize_rows(rows, model_path, stats_data_paths, eval_data_paths, shield_params):
    total = len(rows)
    collision_count = sum(int(bool(row["collision"])) for row in rows)
    merge_success = sum(int(bool(row["merge_success"])) for row in rows)
    endpoint_success = sum(int(bool(row["endpoint_success"])) for row in rows)
    safety_success = sum(int(bool(row["safety_success"])) for row in rows)
    timeout_or_stuck = sum(1 for row in rows if not bool(row["endpoint_success"]))
    total_steps = max(sum(int(row["steps"]) for row in rows), 1)
    total_interventions = sum(int(row["shield_interventions"]) for row in rows)
    total_warnings = sum(int(row["shield_warnings"]) for row in rows)

    report = {
        "model_path": model_path,
        "stats_data_paths": stats_data_paths,
        "eval_data_paths": eval_data_paths,
        "shield_params": shield_params,
        "episodes_evaluated": total,
        "merge_success_rate": merge_success / total if total else 0.0,
        "endpoint_success_rate": endpoint_success / total if total else 0.0,
        "safety_success_rate": safety_success / total if total else 0.0,
        "collision_rate": collision_count / total if total else 0.0,
        "collision_count": collision_count,
        "timeout_or_stuck_rate": timeout_or_stuck / total if total else 0.0,
        "shield_interventions_total": total_interventions,
        "shield_intervention_rate": total_interventions / total_steps,
        "episodes_with_shield_intervention": sum(int(row["shield_interventions"] > 0) for row in rows),
        "shield_warnings_total": total_warnings,
        "shield_warning_rate": total_warnings / total_steps,
        "episodes_with_shield_warning": sum(int(row["shield_warnings"] > 0) for row in rows),
        "mean_episode_reward": float(np.mean([row["episode_reward"] for row in rows])) if rows else 0.0,
        "mean_eval_dense_return": float(np.mean([row["eval_dense_return"] for row in rows])) if rows else 0.0,
        "mean_paper_score": float(np.mean([row["paper_score"] for row in rows])) if rows else 0.0,
        "mean_steps": float(np.mean([row["steps"] for row in rows])) if rows else 0.0,
        "mean_speed_mps": float(np.mean([row["mean_speed_mps"] for row in rows])) if rows else 0.0,
        "mean_abs_jerk_mps3": float(np.mean([row["mean_abs_jerk_mps3"] for row in rows])) if rows else 0.0,
        "mean_min_ttc": float(np.mean([row["min_ttc"] for row in rows])) if rows else 0.0,
        "mean_min_thw": float(np.mean([row["min_thw"] for row in rows])) if rows else 0.0,
    }
    return report


def print_report(report, title="I80 TRANSFER SHIELD EVALUATION REPORT"):
    print("\n" + "=" * 76)
    print(f"{title} (N={report['episodes_evaluated']})")
    print("=" * 76)
    print(f"{'Metric':<32} | {'Value':<15} | {'Unit':<10}")
    print("-" * 76)
    print(f"{'Merge Success Rate':<32} | {report['merge_success_rate']:.3f}           | -")
    print(f"{'Endpoint Success Rate':<32} | {report['endpoint_success_rate']:.3f}           | -")
    print(f"{'Safety Success Rate':<32} | {report['safety_success_rate']:.3f}           | -")
    print(f"{'Collision Rate':<32} | {report['collision_rate']:.3f}           | -")
    print(f"{'Timeout/Stuck Rate':<32} | {report['timeout_or_stuck_rate']:.3f}           | -")
    print("-" * 76)
    print(f"{'Shield Intervention Rate':<32} | {report['shield_intervention_rate']:.3f}           | -")
    print(f"{'Shield Warning Rate':<32} | {report['shield_warning_rate']:.3f}           | -")
    print("-" * 76)
    print(f"{'Avg Speed':<32} | {report['mean_speed_mps']:.3f}           | m/s")
    print(f"{'Avg Jerk':<32} | {report['mean_abs_jerk_mps3']:.3f}           | m/s^3")
    print(f"{'Avg TTC (Min)':<32} | {report['mean_min_ttc']:.3f}           | s")
    print(f"{'Avg THW (Min)':<32} | {report['mean_min_thw']:.3f}           | s")
    print("=" * 76)


def default_output_paths():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        os.path.join(root_dir, "train_log", f"i80_transfer_eval_v10b_U220_D230_epoch290_{stamp}.json"),
        os.path.join(root_dir, "train_log", f"i80_transfer_eval_v10b_U220_D230_epoch290_{stamp}_details.csv"),
    )


def parse_args():
    default_json, default_csv = default_output_paths()
    parser = argparse.ArgumentParser(description="Evaluate I80 transfer performance with v10b shield.")
    parser.add_argument("--model-path", default=build_default_model_path())
    parser.add_argument("--stats-data-path", nargs="+", default=build_default_stats_paths())
    parser.add_argument("--eval-data-path", nargs="+", required=True)
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--save-json", default=default_json)
    parser.add_argument("--save-csv", default=default_csv)
    parser.add_argument("--shield-variant", default="v10b_leadgap_policy_veto_recovery")
    parser.add_argument("--prediction-horizon", type=float, default=1.0)
    parser.add_argument("--lead-ttc-min", type=float, default=3.0)
    parser.add_argument("--follow-ttc-min", type=float, default=3.0)
    parser.add_argument("--lead-thw-min", type=float, default=1.0)
    parser.add_argument("--follow-thw-min", type=float, default=0.8)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()

    _, (expert_mean, expert_std) = load_stats_dataset(args.stats_data_path, device=device)
    dataset = load_eval_dataset_with_fixed_stats(
        args.eval_data_path,
        expert_mean=expert_mean,
        expert_std=expert_std,
        device=device,
    )

    model = PPO.load(args.model_path, device=device)
    shield_params = build_shield_params(
        variant=args.shield_variant,
        prediction_horizon=args.prediction_horizon,
        lead_ttc_min=args.lead_ttc_min,
        follow_ttc_min=args.follow_ttc_min,
        lead_thw_min=args.lead_thw_min,
        follow_thw_min=args.follow_thw_min,
    )

    indices = list(range(len(dataset)))
    if args.num_episodes is not None and args.num_episodes < len(indices):
        rng = np.random.default_rng(args.seed)
        indices = sorted(rng.choice(indices, size=args.num_episodes, replace=False).tolist())

    rows = []
    for traj_index in tqdm(indices, desc="I80 transfer shield eval"):
        row = evaluate_single_trajectory_with_shield(
            model=model,
            dataset=dataset,
            cfg=cfg,
            traj_index=traj_index,
            shield_params=shield_params,
        )
        rows.append(row)

    report = summarize_rows(
        rows=rows,
        model_path=args.model_path,
        stats_data_paths=args.stats_data_path,
        eval_data_paths=args.eval_data_path,
        shield_params=shield_params,
    )
    print_report(report)

    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json), exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Saved report: {args.save_json}")

    if args.save_csv:
        os.makedirs(os.path.dirname(args.save_csv), exist_ok=True)
        pd.DataFrame(rows).to_csv(args.save_csv, index=False, encoding="utf-8-sig")
        print(f"Saved details: {args.save_csv}")


if __name__ == "__main__":
    main()
