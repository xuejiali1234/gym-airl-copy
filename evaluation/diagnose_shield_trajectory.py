import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from configs.config import Config
from envs.merging_env import MergingEnv
from evaluation.failure_case_full_evaluate import TARGET_MODELS, SingleTrajDataset, load_dataset
from evaluation.safety_shield_evaluate import (
    V6_VARIANTS,
    V7_VARIANTS,
    V8_VARIANTS,
    V9_VARIANTS,
    V10_VARIANTS,
    action_risk,
    candidate_actions,
    shield_action,
)


DEFAULT_TRAJECTORY = "vehicle_1085_trajectory.csv"
DEFAULT_MODEL_TAG = "U220_D230_epoch290"
DEFAULT_VARIANTS = [
    "policy",
    "v8a_follow_burst",
    "v9b_near_aux_recovery",
]


TRACE_FIELDS = [
    "variant",
    "step",
    "time_s",
    "px",
    "py",
    "vx",
    "vy",
    "lane_region",
    "policy_action_x",
    "policy_action_y",
    "action_x",
    "action_y",
    "shield_intervened",
    "shield_warning",
    "shield_candidate",
    "shield_reason",
    "original_risk",
    "selected_risk",
    "risk_vehicle",
    "is_critical",
    "predicted_collision",
    "risk_min_ttc",
    "risk_min_thw",
    "l5_lead_dx",
    "l5_lead_dy",
    "l5_lead_rel_vy",
    "l5_lead_gap",
    "l5_lead_ttc",
    "l5_lead_thw",
    "l5_follow_dx",
    "l5_follow_dy",
    "l5_follow_rel_vy",
    "l5_follow_gap",
    "l5_follow_ttc",
    "l5_follow_thw",
    "eval_min_ttc",
    "eval_min_thw",
    "collided",
    "merge_success",
    "endpoint_success",
    "safety_success",
]

CANDIDATE_FIELDS = [
    "variant",
    "step",
    "time_s",
    "candidate",
    "action_x",
    "action_y",
    "risk",
    "unsafe",
    "collided",
    "min_ttc",
    "min_thw",
    "nearest",
    "critical",
    "critical_vehicle",
]


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def get_model_info(tag):
    for item in TARGET_MODELS:
        if item["tag"] == tag:
            return item
    raise ValueError(f"Unknown model tag: {tag}")


def find_traj_matches(dataset, filename):
    matches = []
    for idx in range(len(dataset)):
        traj = dataset[idx]
        if traj.get("filename") == filename:
            matches.append(idx)
    return matches


def lane_region(px, cfg):
    lane_divider_x = cfg.X_MIN + cfg.LANE_WIDTH
    if px < lane_divider_x - 3.28:
        return "target"
    if px > lane_divider_x:
        return "aux"
    return "transition"


def lead_metrics(px, py, vy, surr, cfg, base):
    ox = float(surr[base])
    oy = float(surr[base + 1])
    ovy = float(surr[base + 2])
    if ox == 0.0 and oy == 0.0:
        return {"dx": "", "dy": "", "rel_vy": "", "gap": "", "ttc": "", "thw": ""}
    gap = (oy - cfg.VEHICLE_LENGTH / 2.0) - (py + cfg.VEHICLE_LENGTH / 2.0)
    rel_v = vy - ovy
    ttc = gap / rel_v if gap > 0.0 and rel_v > 0.1 else 20.0
    thw = gap / max(vy, 1e-3) if gap > 0.0 else 0.0
    return {
        "dx": float(ox - px),
        "dy": float(oy - py),
        "rel_vy": float(ovy - vy),
        "gap": float(gap),
        "ttc": float(ttc),
        "thw": float(thw),
    }


def follow_metrics(px, py, vy, surr, cfg, base):
    ox = float(surr[base])
    oy = float(surr[base + 1])
    ovy = float(surr[base + 2])
    if ox == 0.0 and oy == 0.0:
        return {"dx": "", "dy": "", "rel_vy": "", "gap": "", "ttc": "", "thw": ""}
    gap = (py - cfg.VEHICLE_LENGTH / 2.0) - (oy + cfg.VEHICLE_LENGTH / 2.0)
    rel_v = ovy - vy
    ttc = gap / rel_v if gap > 0.0 and rel_v > 0.1 else 20.0
    thw = gap / max(ovy, 1e-3) if gap > 0.0 else 0.0
    return {
        "dx": float(ox - px),
        "dy": float(oy - py),
        "rel_vy": float(ovy - vy),
        "gap": float(gap),
        "ttc": float(ttc),
        "thw": float(thw),
    }


def candidate_risk_fields(score):
    return {
        "risk": float(score["risk"]),
        "unsafe": bool(score["unsafe"]),
        "collided": bool(score["collided"]),
        "min_ttc": float(score["min_ttc"]),
        "min_thw": float(score["min_thw"]),
        "nearest": score.get("nearest", ""),
        "critical": bool(score.get("critical", False)),
        "critical_vehicle": score.get("critical_vehicle", ""),
    }


def make_shield_params(variant):
    margin_variants = (
        "v3_gate_margin",
        "v4_combo",
        "v5_critical_override",
        *V6_VARIANTS,
        *V7_VARIANTS,
        *V8_VARIANTS,
        *V9_VARIANTS,
        *V10_VARIANTS,
    )
    risk_improvement_margin = 0.0
    if variant in margin_variants:
        risk_improvement_margin = 0.20 if variant == "v6c_margin020" else 0.15

    max_intervention_rate = 1.0
    if variant in (*V7_VARIANTS, *V8_VARIANTS, *V9_VARIANTS, *V10_VARIANTS):
        max_intervention_rate = 0.60
    elif variant in ("v5_critical_override", *V6_VARIANTS):
        max_intervention_rate = 0.55
    elif variant == "v4_combo":
        max_intervention_rate = 0.50

    recovery_enabled = variant in (
        "v5_critical_override",
        "v6b_recovery_merge",
        "v6c_margin020",
        *V7_VARIANTS,
        *V8_VARIANTS,
        *V9_VARIANTS,
        *V10_VARIANTS,
    )
    recovery_safe_steps = 3 if variant in ("v6b_recovery_merge", "v6c_margin020") else 5
    recovery_policy_steps = 6 if variant in ("v6b_recovery_merge", "v6c_margin020") else 8

    shield_params = {
        "shield_variant": variant,
        "prediction_horizon": 1.0,
        "lead_ttc_min": 3.0,
        "follow_ttc_min": 3.0,
        "lead_thw_min": 1.0,
        "follow_thw_min": 0.8,
        "risk_improvement_margin": risk_improvement_margin,
        "max_intervention_rate": max_intervention_rate,
        "max_consecutive_interventions": (
            10
            if variant in ("v4_combo", "v5_critical_override", *V6_VARIANTS, *V7_VARIANTS, *V8_VARIANTS, *V9_VARIANTS, *V10_VARIANTS)
            else 10 ** 9
        ),
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
    }
    if variant == "v9b_near_aux_recovery":
        shield_params.update({
            "merge_recovery_aux_slack": 2.0,
            "merge_recovery_min_progress": 0.15,
            "merge_recovery_target_x": -0.20,
            "merge_recovery_x_options": (-1.0, -0.75, -0.50, -0.35, -0.25, -0.15, -0.10, -0.05),
            "merge_recovery_risk_slack": 1.50,
        })
    if variant == "v10a_policy_veto_recovery":
        shield_params.update({
            "merge_recovery_aux_slack": 2.0,
            "merge_recovery_min_progress": 0.15,
            "merge_recovery_target_x": -0.20,
            "merge_recovery_x_options": (-1.0, -0.75, -0.50, -0.35, -0.25, -0.15, -0.10, -0.05),
            "merge_recovery_risk_slack": 1.50,
            "merge_recovery_policy_veto_x": 0.25,
        })
    if variant == "v10b_leadgap_policy_veto_recovery":
        shield_params.update({
            "merge_recovery_aux_slack": 2.0,
            "merge_recovery_min_progress": 0.15,
            "merge_recovery_target_x": -0.20,
            "merge_recovery_x_options": (-1.0, -0.75, -0.50, -0.35, -0.25, -0.15, -0.10, -0.05),
            "merge_recovery_risk_slack": 1.50,
            "merge_recovery_policy_veto_x": 0.25,
            "merge_recovery_policy_veto_lead_gap": 100.0,
            "merge_recovery_policy_veto_lead_thw": 5.0,
        })
    return shield_params


def trace_variant(model, dataset, cfg, traj_index, variant):
    traj = dataset[traj_index]
    env = MergingEnv(SingleTrajDataset(traj, dataset.expert_mean, dataset.expert_std))
    env.collision_margin = 1.0
    obs, _ = env.reset(seed=cfg.SEED + traj_index)
    shield_params = make_shield_params("v5_critical_override" if variant == "policy" else variant)

    rows = []
    candidate_rows = []
    terminated = False
    truncated = False
    step = 0
    max_steps = len(env.current_traj["ego_pos"]) + 50
    shield_interventions = 0
    consecutive_interventions = 0
    safe_steps = 0

    while not (terminated or truncated) and step < max_steps:
        policy_action, _ = model.predict(obs, deterministic=True)
        policy_action = np.asarray(policy_action, dtype=np.float32).reshape(2)
        original_score = action_risk(env, policy_action, policy_action, shield_params)

        if variant == "policy":
            action = policy_action
            shield_info = {
                "intervened": False,
                "warning": False,
                "candidate": "policy",
                "reason": "policy_only",
                "original_risk": original_score["risk"],
                "selected_risk": original_score["risk"],
            }
        else:
            action, shield_info = shield_action(
                env=env,
                policy_action=policy_action,
                params=shield_params,
                step_count=step,
                shield_interventions=shield_interventions,
                consecutive_interventions=consecutive_interventions,
                safe_steps=safe_steps,
            )

        risk_vehicle = original_score.get("critical_vehicle") or original_score.get("nearest", "")
        candidates = candidate_actions(
            policy_action,
            float(env.ego_state[0]),
            env.cfg,
            variant=shield_params["shield_variant"],
            risk_vehicle=risk_vehicle,
            critical=bool(original_score.get("critical", False)),
        )
        candidate_rows.append({
            "variant": variant,
            "step": step,
            "time_s": step * cfg.DT,
            "candidate": "policy",
            "action_x": float(policy_action[0]),
            "action_y": float(policy_action[1]),
            **candidate_risk_fields(original_score),
        })
        for name, cand in candidates:
            if name == "policy":
                continue
            score = action_risk(env, cand, policy_action, shield_params)
            candidate_rows.append({
                "variant": variant,
                "step": step,
                "time_s": step * cfg.DT,
                "candidate": name,
                "action_x": float(cand[0]),
                "action_y": float(cand[1]),
                **candidate_risk_fields(score),
            })

        surr = env._get_surround_at_t(env.t)
        px, py, vx, vy = [float(x) for x in env.ego_state]
        lead = lead_metrics(px, py, vy, surr, cfg, 4)
        follow = follow_metrics(px, py, vy, surr, cfg, 8)

        obs, _, terminated, truncated, info = env.step(action)

        if shield_info["intervened"]:
            shield_interventions += 1
            consecutive_interventions += 1
        else:
            consecutive_interventions = 0

        collided_now = bool(info.get("is_collided", False))
        min_ttc = float(info.get("eval_min_ttc", 20.0))
        min_thw = float(info.get("eval_min_thw", 10.0))
        if not collided_now and min_ttc >= shield_params["lead_ttc_min"] and min_thw >= shield_params["follow_thw_min"]:
            safe_steps += 1
        else:
            safe_steps = 0

        rows.append({
            "variant": variant,
            "step": step,
            "time_s": step * cfg.DT,
            "px": px,
            "py": py,
            "vx": vx,
            "vy": vy,
            "lane_region": lane_region(px, cfg),
            "policy_action_x": float(policy_action[0]),
            "policy_action_y": float(policy_action[1]),
            "action_x": float(action[0]),
            "action_y": float(action[1]),
            "shield_intervened": bool(shield_info["intervened"]),
            "shield_warning": bool(shield_info.get("warning", False)),
            "shield_candidate": shield_info["candidate"],
            "shield_reason": shield_info["reason"],
            "original_risk": shield_info["original_risk"],
            "selected_risk": shield_info["selected_risk"],
            "risk_vehicle": risk_vehicle,
            "is_critical": bool(original_score.get("critical", False)),
            "predicted_collision": bool(original_score.get("collided", False)),
            "risk_min_ttc": float(original_score.get("min_ttc", 20.0)),
            "risk_min_thw": float(original_score.get("min_thw", 10.0)),
            "l5_lead_dx": lead["dx"],
            "l5_lead_dy": lead["dy"],
            "l5_lead_rel_vy": lead["rel_vy"],
            "l5_lead_gap": lead["gap"],
            "l5_lead_ttc": lead["ttc"],
            "l5_lead_thw": lead["thw"],
            "l5_follow_dx": follow["dx"],
            "l5_follow_dy": follow["dy"],
            "l5_follow_rel_vy": follow["rel_vy"],
            "l5_follow_gap": follow["gap"],
            "l5_follow_ttc": follow["ttc"],
            "l5_follow_thw": follow["thw"],
            "eval_min_ttc": min_ttc,
            "eval_min_thw": min_thw,
            "collided": collided_now,
            "merge_success": bool(info.get("is_merge_success", False)),
            "endpoint_success": bool(info.get("is_endpoint_success", False)),
            "safety_success": bool(info.get("is_safety_success", False)),
        })
        step += 1

    return rows, candidate_rows


def summarize_variant(rows):
    collision_rows = [row for row in rows if row["collided"]]
    first_collision = collision_rows[0] if collision_rows else None
    interventions = [row for row in rows if row["shield_intervened"]]
    first_intervention = interventions[0] if interventions else None
    warnings = [row for row in rows if row["shield_warning"]]
    first_warning = warnings[0] if warnings else None

    def min_nonempty(field):
        values = [float(row[field]) for row in rows if row[field] != ""]
        return min(values) if values else ""

    return {
        "steps": len(rows),
        "first_collision_step": first_collision["step"] if first_collision else "",
        "first_collision_time_s": first_collision["time_s"] if first_collision else "",
        "first_collision_candidate": first_collision["shield_candidate"] if first_collision else "",
        "first_intervention_step": first_intervention["step"] if first_intervention else "",
        "first_intervention_candidate": first_intervention["shield_candidate"] if first_intervention else "",
        "first_intervention_reason": first_intervention["shield_reason"] if first_intervention else "",
        "first_warning_step": first_warning["step"] if first_warning else "",
        "first_warning_reason": first_warning["shield_reason"] if first_warning else "",
        "shield_intervention_rate": float(len(interventions) / max(len(rows), 1)),
        "shield_warning_rate": float(len(warnings) / max(len(rows), 1)),
        "min_l5_follow_ttc": min_nonempty("l5_follow_ttc"),
        "min_l5_follow_thw": min_nonempty("l5_follow_thw"),
        "min_eval_ttc": min(float(row["eval_min_ttc"]) for row in rows) if rows else "",
        "min_eval_thw": min(float(row["eval_min_thw"]) for row in rows) if rows else "",
        "final_lane_region": rows[-1]["lane_region"] if rows else "",
        "final_px": rows[-1]["px"] if rows else "",
        "merge_success": rows[-1]["merge_success"] if rows else "",
        "endpoint_success": rows[-1]["endpoint_success"] if rows else "",
        "safety_success": rows[-1]["safety_success"] if rows else "",
    }


def write_summary_md(output_dir, trajectory, model_info, summaries):
    lines = [
        "# Shield trajectory diagnosis",
        "",
        f"Trajectory: `{trajectory}`",
        f"Model: `{model_info['tag']}`",
        f"Checkpoint: `{model_info['checkpoint']}`",
        "",
        "| Variant | Merge | Endpoint | Safety | Min TTC | Min THW | First Collision | First Intervention | Shield Rate | Final Lane | Final Px |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|---|---:|",
    ]
    for item in summaries:
        lines.append(
            "| {variant} | {merge} | {endpoint} | {safety} | {min_ttc:.3f} | {min_thw:.3f} | {collision} | {intervention} | {rate:.3f} | {lane} | {px:.3f} |".format(
                variant=item["variant"],
                merge=int(bool(item["merge_success"])),
                endpoint=int(bool(item["endpoint_success"])),
                safety=int(bool(item["safety_success"])),
                min_ttc=float(item["min_eval_ttc"]) if item["min_eval_ttc"] != "" else float("nan"),
                min_thw=float(item["min_eval_thw"]) if item["min_eval_thw"] != "" else float("nan"),
                collision=item["first_collision_step"] if item["first_collision_step"] != "" else "-",
                intervention=item["first_intervention_candidate"] or "-",
                rate=float(item["shield_intervention_rate"]),
                lane=item["final_lane_region"] or "-",
                px=float(item["final_px"]) if item["final_px"] != "" else float("nan"),
            )
        )
    (output_dir / "diagnosis_summary.md").write_text("\n".join(lines), encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", default=DEFAULT_TRAJECTORY)
    parser.add_argument("--traj-index", type=int, default=None)
    parser.add_argument("--model-tag", default=DEFAULT_MODEL_TAG)
    parser.add_argument("--variant", action="append", default=[])
    parser.add_argument(
        "--output-dir",
        default=str(ROOT_DIR / "train_log" / "shield_trajectory_diagnosis"),
    )
    args = parser.parse_args()

    variants = args.variant or DEFAULT_VARIANTS
    traj_stem = Path(args.trajectory).stem
    output_dir = Path(args.output_dir).resolve() / traj_stem
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()
    dataset = load_dataset(device)
    if args.traj_index is not None:
        traj_index = args.traj_index
        if not (0 <= traj_index < len(dataset)):
            raise ValueError(f"traj_index out of range: {traj_index}")
        actual_name = dataset[traj_index].get("filename")
        if actual_name != args.trajectory:
            raise ValueError(
                f"traj_index {traj_index} points to {actual_name!r}, not {args.trajectory!r}"
            )
    else:
        matches = find_traj_matches(dataset, args.trajectory)
        if not matches:
            raise ValueError(f"Trajectory not found: {args.trajectory}")
        if len(matches) > 1:
            raise ValueError(
                f"Trajectory filename {args.trajectory!r} is ambiguous; matching traj_index values: {matches}"
            )
        traj_index = matches[0]
    model_info = get_model_info(args.model_tag)
    model = PPO.load(str(model_info["checkpoint"]), device=device)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "trajectory": args.trajectory,
        "traj_index": traj_index,
        "model_tag": args.model_tag,
        "checkpoint": str(model_info["checkpoint"]),
        "variants": variants,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8-sig",
    )

    all_summaries = []
    for variant in variants:
        rows, candidate_rows = trace_variant(model, dataset, cfg, traj_index, variant)
        write_csv(output_dir / f"{variant}_{traj_stem}_trace.csv", rows, TRACE_FIELDS)
        write_csv(output_dir / f"{variant}_{traj_stem}_candidate_risks.csv", candidate_rows, CANDIDATE_FIELDS)
        all_summaries.append({"variant": variant, **summarize_variant(rows)})

    summary_fields = list(all_summaries[0].keys()) if all_summaries else []
    write_csv(output_dir / f"{traj_stem}_variant_summary.csv", all_summaries, summary_fields)
    write_summary_md(output_dir, args.trajectory, model_info, all_summaries)
    print(f"Saved trajectory diagnosis to: {output_dir}")


if __name__ == "__main__":
    main()
