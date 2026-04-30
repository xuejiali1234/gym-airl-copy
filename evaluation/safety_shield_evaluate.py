import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from configs.config import Config
from envs.merging_env import MergingEnv
from evaluation.failure_case_full_evaluate import (  # noqa: E402
    DETAIL_FIELDS,
    TARGET_MODELS,
    SingleTrajDataset,
    classify_failure,
    load_dataset,
    nearest_vehicle_snapshot,
    paper_score,
    summarize_rows,
    write_csv,
    write_overlap,
)
from evaluation.hard_case_protocol_evaluate import (  # noqa: E402
    DEFAULT_HARD_CASE_LIST,
    PROTOCOL_SUMMARY_FIELDS,
    annotate_summary,
    dataset_indices_by_filename,
    load_hard_case_filenames,
    parse_model_arg,
)


SHIELD_DETAIL_FIELDS = DETAIL_FIELDS + [
    "shield_enabled",
    "shield_variant",
    "shield_interventions",
    "shield_intervention_rate",
    "shield_warnings",
    "shield_warning_rate",
    "most_common_shield_candidate",
    "hold_lane_count",
    "delay_lateral_count",
    "hard_brake_hold_count",
    "first_merge_step",
    "first_merge_time_s",
    "final_lane",
    "first_shield_step",
    "first_shield_time_s",
    "first_shield_reason",
    "first_shield_candidate",
    "first_shield_original_action_x",
    "first_shield_original_action_y",
    "first_shield_action_x",
    "first_shield_action_y",
    "first_shield_original_risk",
    "first_shield_selected_risk",
    "first_shield_warning_step",
    "first_shield_warning_time_s",
    "first_shield_warning_reason",
]

SHIELD_SUMMARY_FIELDS = PROTOCOL_SUMMARY_FIELDS + [
    "shield_interventions_total",
    "shield_intervention_rate",
    "episodes_with_shield_intervention",
    "shield_warnings_total",
    "shield_warning_rate",
    "episodes_with_shield_warning",
]

MERGE_FALSE_SUMMARY_FIELDS = [
    "model_tag",
    "split",
    "total",
    "merge_false_count",
    "merge_false_rate",
    "endpoint_true_merge_false_count",
    "endpoint_true_merge_false_rate",
    "endpoint_false_merge_false_count",
    "safety_false_merge_false_count",
    "collision_merge_false_count",
    "mean_shield_intervention_rate",
    "mean_dist_to_goal",
    "final_target_count",
    "final_transition_count",
    "final_aux_count",
    "most_common_shield_candidate",
    "most_common_first_shield_candidate",
    "mean_hold_lane_count",
    "mean_delay_lateral_count",
    "mean_hard_brake_hold_count",
    "endpoint_true_merge_false_filenames",
]

V6_VARIANTS = (
    "v6a_follow_emergency",
    "v6b_recovery_merge",
    "v6c_margin020",
)

V7_VARIANTS = (
    "v7a_follow_low_thw",
    "v7b_merge_preserving_warning",
)

V8_VARIANTS = (
    "v8a_follow_burst",
)

V9_VARIANTS = (
    "v9a_merge_recovery",
    "v9b_near_aux_recovery",
)

V10_VARIANTS = (
    "v10a_policy_veto_recovery",
    "v10b_leadgap_policy_veto_recovery",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluation-only target-lane safety shield. "
            "This script never trains, never modifies rewards, and never changes checkpoint saving."
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
        help="CSV containing the fixed hard-case filenames.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory. Defaults to train_log/safety_shield_eval_<timestamp>.",
    )
    parser.add_argument(
        "--include-full",
        action="store_true",
        help="Also evaluate the full 217-trajectory split. Default evaluates hard-15 only.",
    )
    parser.add_argument("--prediction-horizon", type=float, default=1.0)
    parser.add_argument("--lead-ttc-min", type=float, default=3.0)
    parser.add_argument("--follow-ttc-min", type=float, default=3.0)
    parser.add_argument("--lead-thw-min", type=float, default=1.0)
    parser.add_argument("--follow-thw-min", type=float, default=0.8)
    parser.add_argument(
        "--shield-variant",
        choices=[
            "default",
            "v1_gate",
            "v2_gate_emergency",
            "v3_gate_margin",
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
        ],
        default="default",
        help=(
            "Shield variant. default keeps the original shield behavior; "
            "v1-v6 are the hard-case refinement probes."
        ),
    )
    parser.add_argument("--endpoint-threshold", type=float, default=0.95)
    parser.add_argument("--safety-threshold", type=float, default=0.90)
    parser.add_argument("--collision-threshold", type=float, default=0.01)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


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


def candidate_actions(policy_action, px, cfg, variant="default", risk_vehicle="", critical=False):
    action = np.clip(np.asarray(policy_action, dtype=np.float32).reshape(2), -1.0, 1.0)
    ax, ay = float(action[0]), float(action[1])
    lane_divider_x = cfg.X_MIN + cfg.LANE_WIDTH
    in_aux_or_near_aux = px >= lane_divider_x - 3.28

    if ax < 0.0:
        delayed_ax = 0.25 * ax
    else:
        delayed_ax = 0.50 * ax

    retreat_ax = 0.20 if in_aux_or_near_aux else 0.0

    if variant in (*V7_VARIANTS, *V8_VARIANTS, *V9_VARIANTS, *V10_VARIANTS) and risk_vehicle == "L5_follow":
        candidates = [
            ("policy", np.array([ax, ay], dtype=np.float32)),
            ("follow_hold_lane", np.array([0.0, ay], dtype=np.float32)),
            ("follow_delay_merge", np.array([delayed_ax, ay], dtype=np.float32)),
            ("follow_speed_match", np.array([0.0, -0.05], dtype=np.float32)),
            ("follow_mild_accel_if_lead_safe", np.array([delayed_ax, max(ay, 0.20)], dtype=np.float32)),
            ("delay_lateral", np.array([delayed_ax, ay], dtype=np.float32)),
            ("speed_match_hold", np.array([0.0, -0.05], dtype=np.float32)),
        ]
        return [(name, np.clip(cand, -1.0, 1.0)) for name, cand in candidates]

    if variant in V6_VARIANTS and critical:
        if risk_vehicle == "L5_follow":
            candidates = [
                ("follow_hold_lane", np.array([0.0, ay], dtype=np.float32)),
                ("follow_delay_merge", np.array([delayed_ax, ay], dtype=np.float32)),
                ("follow_speed_match", np.array([0.0, -0.05], dtype=np.float32)),
                ("follow_mild_accel_if_lead_safe", np.array([delayed_ax, max(ay, 0.20)], dtype=np.float32)),
            ]
        else:
            candidates = [
                ("lead_hold_lane", np.array([0.0, min(ay, 0.0)], dtype=np.float32)),
                ("lead_delay_lateral", np.array([delayed_ax, min(ay, 0.0)], dtype=np.float32)),
                ("lead_hold_lane_mild_decel", np.array([0.0, min(ay, -0.30)], dtype=np.float32)),
                ("lead_hold_lane_strong_decel", np.array([0.0, -0.85], dtype=np.float32)),
                ("lead_abort_merge", np.array([max(retreat_ax, 0.35), -0.60], dtype=np.float32)),
                ("lead_abort_merge_full_brake", np.array([max(retreat_ax, 0.45), -1.0], dtype=np.float32)),
            ]
        return [(name, np.clip(cand, -1.0, 1.0)) for name, cand in candidates]

    candidates = [
        ("policy", np.array([ax, ay], dtype=np.float32)),
        ("delay_lateral", np.array([delayed_ax, ay], dtype=np.float32)),
        ("hold_lane", np.array([0.0, min(ay, 0.0)], dtype=np.float32)),
        ("mild_brake_hold", np.array([0.0, min(ay, -0.20)], dtype=np.float32)),
        ("hard_brake_hold", np.array([0.0, -0.60], dtype=np.float32)),
        ("brake_retreat_aux", np.array([retreat_ax, -0.30], dtype=np.float32)),
        ("speed_match_hold", np.array([0.0, -0.05], dtype=np.float32)),
    ]
    if variant in ("v2_gate_emergency", "v4_combo"):
        candidates.extend([
            ("hold_lane_strong_decel", np.array([0.0, -0.85], dtype=np.float32)),
            ("hold_lane_full_brake", np.array([0.0, -1.0], dtype=np.float32)),
            ("abort_merge_strong_decel", np.array([max(retreat_ax, 0.35), -0.85], dtype=np.float32)),
            ("abort_merge_full_brake", np.array([max(retreat_ax, 0.45), -1.0], dtype=np.float32)),
        ])
    elif variant in ("v5_critical_override", *V7_VARIANTS, *V8_VARIANTS, *V9_VARIANTS, *V10_VARIANTS):
        if risk_vehicle == "L5_follow":
            candidates.extend([
                ("follow_hold_lane", np.array([0.0, ay], dtype=np.float32)),
                ("follow_delay_merge", np.array([delayed_ax, ay], dtype=np.float32)),
                ("follow_speed_match", np.array([0.0, -0.05], dtype=np.float32)),
                ("follow_mild_accel_if_lead_safe", np.array([delayed_ax, max(ay, 0.20)], dtype=np.float32)),
            ])
        else:
            candidates.extend([
                ("lead_hold_lane", np.array([0.0, min(ay, 0.0)], dtype=np.float32)),
                ("lead_delay_lateral", np.array([delayed_ax, min(ay, 0.0)], dtype=np.float32)),
                ("lead_hold_lane_mild_decel", np.array([0.0, min(ay, -0.30)], dtype=np.float32)),
                ("lead_hold_lane_strong_decel", np.array([0.0, -0.85], dtype=np.float32)),
                ("lead_abort_merge", np.array([max(retreat_ax, 0.35), -0.60], dtype=np.float32)),
                ("lead_abort_merge_full_brake", np.array([max(retreat_ax, 0.45), -1.0], dtype=np.float32)),
            ])
    return [(name, np.clip(cand, -1.0, 1.0)) for name, cand in candidates]


def predict_next_state(cfg, state, action):
    px, py, vx, vy = [float(x) for x in state]
    ax = float(action[0]) * getattr(cfg, "PHYS_STEER_MAX", 8.0)
    ay = float(action[1]) * getattr(cfg, "PHYS_ACC_MAX", 15.0)
    dt = cfg.DT

    vx_new = vx + ax * dt
    vy_new = vy + ay * dt

    speed_limit = getattr(cfg, "SPEED_LIMIT", 80.0)
    current_speed = float(np.hypot(vx_new, vy_new))
    if current_speed > speed_limit:
        ratio = speed_limit / current_speed
        vx_new *= ratio
        vy_new *= ratio
    if vy_new < 0.0:
        vy_new = 0.0

    px_new = px + vx_new * dt
    py_new = py + vy_new * dt

    half_width = cfg.VEHICLE_WIDTH / 2.0
    wall_min = cfg.X_MIN + half_width
    wall_max = cfg.X_MAX - half_width
    if px_new < wall_min:
        px_new = wall_min
        if vx_new < 0.0:
            vx_new *= 0.1
    elif px_new > wall_max:
        px_new = wall_max
        if vx_new > 0.0:
            vx_new *= 0.1

    return np.array([px_new, py_new, vx_new, vy_new], dtype=np.float32)


def target_lane_soft_overlap_risk(env, px, py, surr_data):
    risk = 0.0
    nearest_name = ""
    nearest_dist = float("inf")
    for base, name in ((4, "L5_lead"), (8, "L5_follow")):
        ox = float(surr_data[base])
        oy = float(surr_data[base + 1])
        if ox == 0.0 and oy == 0.0:
            continue
        dx = abs(px - ox)
        dy = abs(py - oy)
        dist = float(np.hypot(dx, dy))
        if dist < nearest_dist:
            nearest_dist = dist
            nearest_name = name
        lateral = max(0.0, 10.0 - dx) / 10.0
        longitudinal = max(0.0, 25.0 - dy) / 25.0
        risk += 30.0 * lateral * longitudinal
    return risk, nearest_name


def target_lane_critical_snapshot(env, px, py, vy, surr_data, params):
    cfg = env.cfg
    half_len_sum = cfg.VEHICLE_LENGTH
    half_width_sum = cfg.VEHICLE_WIDTH
    best = {
        "critical": False,
        "predicted_overlap": False,
        "vehicle": "",
        "min_center_gap": float("inf"),
        "risk_score": 0.0,
    }
    for base, name in ((4, "L5_lead"), (8, "L5_follow")):
        ox = float(surr_data[base])
        oy = float(surr_data[base + 1])
        ovy = float(surr_data[base + 2])
        if ox == 0.0 and oy == 0.0:
            continue

        dx = abs(px - ox)
        center_gap = abs(py - oy)
        if dx > cfg.LANE_WIDTH:
            continue

        predicted_overlap = dx < half_width_sum and center_gap < half_len_sum
        if name == "L5_lead":
            gap = (oy - cfg.VEHICLE_LENGTH / 2.0) - (py + cfg.VEHICLE_LENGTH / 2.0)
            rel_v = vy - ovy
            thw = gap / max(vy, 1e-3) if gap > 0.0 else 0.0
            ttc = gap / rel_v if gap > 0.0 and rel_v > 0.1 else 20.0
            ttc_limit = params["critical_ttc_min"]
            thw_limit = params["critical_thw_min"]
        else:
            gap = (py - cfg.VEHICLE_LENGTH / 2.0) - (oy + cfg.VEHICLE_LENGTH / 2.0)
            rel_v = ovy - vy
            thw = gap / max(ovy, 1e-3) if gap > 0.0 else 0.0
            ttc = gap / rel_v if gap > 0.0 and rel_v > 0.1 else 20.0
            ttc_limit = params["critical_ttc_min"]
            thw_limit = params["critical_follow_thw_min"]

        tiny_gap = center_gap < cfg.VEHICLE_LENGTH * 0.8
        critical = predicted_overlap or ttc < ttc_limit or thw < thw_limit or tiny_gap
        score = 0.0
        if predicted_overlap:
            score += 1000.0
        score += max(0.0, (ttc_limit - ttc) / max(ttc_limit, 1e-6)) * 100.0
        score += max(0.0, (thw_limit - thw) / max(thw_limit, 1e-6)) * 80.0
        score += max(0.0, (cfg.VEHICLE_LENGTH * 0.8 - center_gap) / max(cfg.VEHICLE_LENGTH * 0.8, 1e-6)) * 60.0

        if critical and (not best["critical"] or score > best["risk_score"]):
            best = {
                "critical": True,
                "predicted_overlap": bool(predicted_overlap),
                "vehicle": name,
                "min_center_gap": float(center_gap),
                "risk_score": float(score),
            }
    return best


def target_lane_vehicle_metrics(env):
    cfg = env.cfg
    px, py, _, vy = [float(x) for x in env.ego_state]
    surr = env._get_surround_at_t(env.t)

    def empty_metrics():
        return {
            "valid": False,
            "gap": float("inf"),
            "ttc": 20.0,
            "thw": 10.0,
            "dx": float("inf"),
            "dy": float("inf"),
            "rel_vy": 0.0,
        }

    def lead_metrics(base):
        ox = float(surr[base])
        oy = float(surr[base + 1])
        ovy = float(surr[base + 2])
        if ox == 0.0 and oy == 0.0:
            return empty_metrics()
        gap = (oy - cfg.VEHICLE_LENGTH / 2.0) - (py + cfg.VEHICLE_LENGTH / 2.0)
        rel_v = vy - ovy
        thw = gap / max(vy, 1e-3) if gap > 0.0 else 0.0
        ttc = gap / rel_v if gap > 0.0 and rel_v > 0.1 else 20.0
        return {
            "valid": True,
            "gap": float(gap),
            "ttc": float(ttc),
            "thw": float(thw),
            "dx": float(ox - px),
            "dy": float(oy - py),
            "rel_vy": float(ovy - vy),
        }

    def follow_metrics(base):
        ox = float(surr[base])
        oy = float(surr[base + 1])
        ovy = float(surr[base + 2])
        if ox == 0.0 and oy == 0.0:
            return empty_metrics()
        gap = (py - cfg.VEHICLE_LENGTH / 2.0) - (oy + cfg.VEHICLE_LENGTH / 2.0)
        rel_v = ovy - vy
        thw = gap / max(ovy, 1e-3) if gap > 0.0 else 0.0
        ttc = gap / rel_v if gap > 0.0 and rel_v > 0.1 else 20.0
        return {
            "valid": True,
            "gap": float(gap),
            "ttc": float(ttc),
            "thw": float(thw),
            "dx": float(ox - px),
            "dy": float(oy - py),
            "rel_vy": float(ovy - vy),
        }

    return {
        "lead": lead_metrics(4),
        "follow": follow_metrics(8),
    }


def action_risk(env, action, policy_action, params):
    cfg = env.cfg
    horizon_steps = max(1, int(round(params["prediction_horizon"] / cfg.DT)))
    state = np.asarray(env.ego_state, dtype=np.float32).copy()
    total_risk = 0.0
    collided = False
    min_ttc_seen = 20.0
    min_thw_seen = 10.0
    nearest_seen = ""
    critical = False
    predicted_overlap = False
    critical_vehicle = ""
    min_center_gap = float("inf")

    for step in range(1, horizon_steps + 1):
        state = predict_next_state(cfg, state, action)
        px, py, _, vy = [float(x) for x in state]
        surr = env._get_surround_at_t(env.t + step)
        step_weight = 1.0 + 0.15 * (horizon_steps - step)

        if env._check_collision(px, py, surr):
            collided = True
            total_risk += step_weight * 10000.0

        min_ttc, min_thw = env._compute_min_ttc_thw(px, py, vy, surr)
        min_ttc_seen = min(min_ttc_seen, min_ttc)
        min_thw_seen = min(min_thw_seen, min_thw)

        ttc_min = params["lead_ttc_min"]
        if min_ttc < ttc_min:
            total_risk += step_weight * 80.0 * ((ttc_min - min_ttc) / max(ttc_min, 1e-6)) ** 2

        thw_min = params["lead_thw_min"]
        if min_thw < thw_min:
            total_risk += step_weight * 60.0 * ((thw_min - min_thw) / max(thw_min, 1e-6)) ** 2

        soft_risk, nearest = target_lane_soft_overlap_risk(env, px, py, surr)
        total_risk += step_weight * soft_risk
        if nearest and not nearest_seen:
            nearest_seen = nearest

        critical_info = target_lane_critical_snapshot(env, px, py, vy, surr, params)
        if critical_info["critical"]:
            critical = True
            predicted_overlap = predicted_overlap or critical_info["predicted_overlap"]
            if not critical_vehicle or critical_info["risk_score"] > 0.0:
                critical_vehicle = critical_info["vehicle"] or critical_vehicle
            min_center_gap = min(min_center_gap, critical_info["min_center_gap"])

    deviation = float(np.sum((np.asarray(action) - np.asarray(policy_action)) ** 2))
    lateral_change = abs(float(action[0] - policy_action[0]))
    accel_change = abs(float(action[1] - policy_action[1]))
    total_risk += 3.0 * deviation + 0.5 * lateral_change + 0.25 * accel_change

    unsafe = collided or min_ttc_seen < params["lead_ttc_min"] or min_thw_seen < params["lead_thw_min"]
    return {
        "risk": float(total_risk),
        "unsafe": bool(unsafe),
        "collided": bool(collided),
        "min_ttc": float(min_ttc_seen),
        "min_thw": float(min_thw_seen),
        "nearest": nearest_seen,
        "critical": bool(critical),
        "predicted_overlap": bool(predicted_overlap or collided),
        "critical_vehicle": critical_vehicle or nearest_seen,
        "min_center_gap": float(min_center_gap) if np.isfinite(min_center_gap) else "",
    }


def is_merge_phase(env, policy_action):
    cfg = env.cfg
    px, _, vx, _ = [float(x) for x in env.ego_state]
    action = np.asarray(policy_action, dtype=np.float32).reshape(2)
    lane_divider_x = cfg.X_MIN + cfg.LANE_WIDTH
    return (
        px <= lane_divider_x + 3.0
        or float(action[0]) < -0.05
        or vx < -0.30
    )


def risk_reason(score, params, prefix=""):
    reason_parts = []
    if score["collided"]:
        reason_parts.append("predicted_collision")
    if score["min_ttc"] < params["lead_ttc_min"]:
        reason_parts.append("low_ttc")
    if score["min_thw"] < params["lead_thw_min"]:
        reason_parts.append("low_thw")
    reason = "+".join(reason_parts) if reason_parts else "risk_reduction"
    return f"{prefix}{reason}" if prefix else reason


def shield_action(
    env,
    policy_action,
    params,
    step_count=0,
    shield_interventions=0,
    consecutive_interventions=0,
    safe_steps=0,
):
    policy_action = np.clip(np.asarray(policy_action, dtype=np.float32).reshape(2), -1.0, 1.0)
    variant = params.get("shield_variant", "default")
    original_score = action_risk(env, policy_action, policy_action, params)
    risk_vehicle = original_score.get("critical_vehicle") or original_score.get("nearest", "")
    is_critical = bool(original_score.get("critical", False))
    candidates = candidate_actions(
        policy_action,
        float(env.ego_state[0]),
        env.cfg,
        variant=variant,
        risk_vehicle=risk_vehicle,
        critical=is_critical,
    )
    scored = [("policy", policy_action, original_score)]
    for name, action in candidates:
        if name == "policy":
            continue
        scored.append((name, action, action_risk(env, action, policy_action, params)))

    original = scored[0]
    merge_phase = is_merge_phase(env, policy_action)
    if not original[2]["unsafe"]:
        return policy_action, {
            "intervened": False,
            "warning": False,
            "candidate": "policy",
            "reason": "safe",
            "original_risk": original[2]["risk"],
            "selected_risk": original[2]["risk"],
        }

    gated_variants = (
        "v1_gate",
        "v2_gate_emergency",
        "v3_gate_margin",
        "v4_combo",
        "v5_critical_override",
        *V6_VARIANTS,
        *V7_VARIANTS,
        *V8_VARIANTS,
        *V9_VARIANTS,
        *V10_VARIANTS,
    )
    if variant in gated_variants and not merge_phase:
        return policy_action, {
            "intervened": False,
            "warning": True,
            "candidate": "policy",
            "reason": risk_reason(original[2], params, prefix="pre_merge_warning_"),
            "original_risk": original[2]["risk"],
            "selected_risk": original[2]["risk"],
        }

    if variant in ("v8a_follow_burst", *V9_VARIANTS, *V10_VARIANTS) and risk_vehicle == "L5_follow":
        pair_metrics = target_lane_vehicle_metrics(env)
        follow = pair_metrics["follow"]
        lead = pair_metrics["lead"]
        follow_emergency = (
            follow["valid"]
            and follow["gap"] > params.get("follow_burst_gap_min", 0.0)
            and (
                follow["thw"] < params.get("follow_burst_thw_trigger", 0.5)
                or follow["ttc"] < params.get("follow_burst_ttc_trigger", 2.0)
            )
        )
        lead_guard_clear = (
            not lead["valid"]
            or (
                lead["gap"] > params.get("follow_burst_lead_gap_min", 15.0)
                and lead["ttc"] > params.get("follow_burst_lead_ttc_min", 1.0)
                and lead["thw"] > params.get("follow_burst_lead_thw_min", 0.30)
            )
        )
        if follow_emergency and lead_guard_clear:
            burst = np.asarray(policy_action, dtype=np.float32).copy()
            if float(burst[0]) < 0.0:
                burst[0] = 0.25 * burst[0]
            else:
                burst[0] = 0.50 * burst[0]
            burst[1] = max(float(burst[1]), params.get("follow_burst_action_y", 1.0))
            burst = np.clip(burst, -1.0, 1.0)
            burst_score = action_risk(env, burst, policy_action, params)
            return burst, {
                "intervened": True,
                "warning": False,
                "candidate": "follow_emergency_burst",
                "reason": (
                    "follow_emergency_burst_"
                    f"thw_{follow['thw']:.3f}_ttc_{follow['ttc']:.3f}"
                ),
                "original_risk": original[2]["risk"],
                "selected_risk": burst_score["risk"],
            }

    max_intervention_rate = params.get("max_intervention_rate", 1.0)
    max_consecutive = params.get("max_consecutive_interventions", 10 ** 9)
    current_rate = shield_interventions / max(step_count, 1)
    if variant in ("v4_combo", "v5_critical_override", *V6_VARIANTS, *V7_VARIANTS, *V8_VARIANTS, *V9_VARIANTS, *V10_VARIANTS) and not is_critical and (
        current_rate >= max_intervention_rate
        or consecutive_interventions >= max_consecutive
    ):
        return policy_action, {
            "intervened": False,
            "warning": True,
            "candidate": "policy",
            "reason": "budget_limit_" + risk_reason(original[2], params),
            "original_risk": original[2]["risk"],
            "selected_risk": original[2]["risk"],
        }

    selected = min(scored, key=lambda item: item[2]["risk"])
    improvement_ratio = (
        (original[2]["risk"] - selected[2]["risk"]) / max(abs(original[2]["risk"]), 1e-6)
    )
    if variant in (*V9_VARIANTS, *V10_VARIANTS):
        cfg = env.cfg
        px, _, vx, _ = [float(x) for x in env.ego_state]
        lane_divider_x = cfg.X_MIN + cfg.LANE_WIDTH
        target_boundary_x = lane_divider_x - 3.28
        aux_slack = params.get("merge_recovery_aux_slack", 0.0)
        traj_len = len(getattr(env, "current_traj", {}).get("ego_pos", []))
        progress_ratio = step_count / max(traj_len, 1)
        in_transition = target_boundary_x <= px <= lane_divider_x + aux_slack
        target_lane_pairs = target_lane_vehicle_metrics(env) if variant in V10_VARIANTS else None
        lead_pair = target_lane_pairs["lead"] if target_lane_pairs else {"valid": False}
        policy_retreat = float(policy_action[0]) > params.get("merge_recovery_policy_veto_x", 0.25)
        policy_retreat_veto = False
        if variant == "v10a_policy_veto_recovery":
            policy_retreat_veto = policy_retreat
        elif variant == "v10b_leadgap_policy_veto_recovery":
            lead_gap = float(lead_pair["gap"]) if lead_pair.get("valid") else float("inf")
            lead_thw = float(lead_pair["thw"]) if lead_pair.get("valid") else float("inf")
            policy_retreat_veto = (
                policy_retreat
                and risk_vehicle == "L5_follow"
                and (
                    lead_gap < params.get("merge_recovery_policy_veto_lead_gap", 100.0)
                    or lead_thw < params.get("merge_recovery_policy_veto_lead_thw", 5.0)
                )
            )
        merge_intent = (
            float(policy_action[0]) < params.get("merge_recovery_policy_x_trigger", -0.03)
            or (vx < -0.05 and not policy_retreat_veto)
        )
        selected_lateral = float(selected[1][0])
        needs_more_lateral = selected_lateral > params.get("merge_recovery_target_x", -0.10)
        if (
            in_transition
            and merge_intent
            and needs_more_lateral
            and progress_ratio >= params.get("merge_recovery_min_progress", 0.35)
        ):
            base_action = np.asarray(selected[1], dtype=np.float32).copy()
            assist_options = []
            for target_x in params.get("merge_recovery_x_options", (-0.15, -0.10, -0.05)):
                assist = base_action.copy()
                assist[0] = min(float(assist[0]), float(target_x))
                assist = np.clip(assist, -1.0, 1.0)
                if abs(float(assist[0]) - float(base_action[0])) < 1e-6:
                    continue
                assist_score = action_risk(env, assist, policy_action, params)
                if assist_score["collided"] or assist_score.get("predicted_overlap", False):
                    continue
                risk_ceiling = max(original[2]["risk"], selected[2]["risk"], 1e-6)
                if assist_score["risk"] <= risk_ceiling * params.get("merge_recovery_risk_slack", 1.05):
                    assist_options.append((float(assist[0]), assist, assist_score))
            if assist_options:
                assist_options.sort(key=lambda item: item[0])
                _, assist, assist_score = assist_options[0]
                return assist, {
                    "intervened": True,
                    "warning": False,
                    "candidate": "merge_recovery_lateral",
                    "reason": "merge_recovery_lateral_" + risk_reason(original[2], params),
                    "original_risk": original[2]["risk"],
                    "selected_risk": assist_score["risk"],
                }

    risk_margin = params.get("risk_improvement_margin", 0.0)
    if variant in ("v3_gate_margin", "v4_combo", "v5_critical_override", *V6_VARIANTS, *V7_VARIANTS, *V8_VARIANTS, *V9_VARIANTS, *V10_VARIANTS) and not is_critical and improvement_ratio < risk_margin:
        return policy_action, {
            "intervened": False,
            "warning": True,
            "candidate": "policy",
            "reason": f"risk_margin_not_met_{risk_reason(original[2], params)}",
            "original_risk": original[2]["risk"],
            "selected_risk": selected[2]["risk"],
        }

    if (
        variant == "v7b_merge_preserving_warning"
        and not is_critical
        and selected[0] != "policy"
        and float(policy_action[0]) < -0.05
    ):
        preserved = np.asarray(selected[1], dtype=np.float32).copy()
        lateral_factor = 0.45 if risk_vehicle == "L5_follow" else 0.35
        target_x = lateral_factor * float(policy_action[0])
        if float(preserved[0]) > target_x:
            preserved[0] = target_x
        preserved = np.clip(preserved, -1.0, 1.0)
        preserved_score = action_risk(env, preserved, policy_action, params)
        if preserved_score["risk"] <= original[2]["risk"]:
            return preserved, {
                "intervened": True,
                "warning": False,
                "candidate": "merge_preserve_lateral",
                "reason": "merge_preserve_lateral_" + risk_reason(original[2], params),
                "original_risk": original[2]["risk"],
                "selected_risk": preserved_score["risk"],
            }

    if (
        variant in ("v6b_recovery_merge", "v6c_margin020")
        and not is_critical
        and params.get("recovery_blend_enabled", False)
        and safe_steps >= params.get("recovery_safe_steps", 3)
        and float(policy_action[0]) < -0.05
    ):
        recovered = np.asarray(selected[1], dtype=np.float32).copy()
        if safe_steps >= params.get("recovery_policy_steps", 6):
            recovered[0] = policy_action[0]
            candidate = "recovery_policy_lateral"
        else:
            recovered[0] = 0.5 * policy_action[0] + 0.5 * selected[1][0]
            candidate = "recovery_blend_lateral"
        recovered[1] = selected[1][1]
        recovered = np.clip(recovered, -1.0, 1.0)
        recovered_score = action_risk(env, recovered, policy_action, params)
        if recovered_score["risk"] <= original[2]["risk"]:
            return recovered, {
                "intervened": True,
                "warning": False,
                "candidate": candidate,
                "reason": candidate + "_" + risk_reason(original[2], params),
                "original_risk": original[2]["risk"],
                "selected_risk": recovered_score["risk"],
            }

    if (
        variant in ("v5_critical_override", *V7_VARIANTS, *V8_VARIANTS, *V9_VARIANTS, *V10_VARIANTS)
        and not is_critical
        and params.get("recovery_blend_enabled", False)
        and safe_steps >= params.get("recovery_safe_steps", 5)
    ):
        if safe_steps >= params.get("recovery_policy_steps", 8):
            return policy_action, {
                "intervened": False,
                "warning": True,
                "candidate": "policy",
                "reason": "recovery_policy_" + risk_reason(original[2], params),
                "original_risk": original[2]["risk"],
                "selected_risk": selected[2]["risk"],
            }
        recovered = 0.7 * policy_action + 0.3 * selected[1]
        recovered = np.clip(recovered, -1.0, 1.0)
        recovered_score = action_risk(env, recovered, policy_action, params)
        if recovered_score["risk"] <= original[2]["risk"]:
            return recovered, {
                "intervened": True,
                "warning": False,
                "candidate": "recovery_blend",
                "reason": "recovery_blend_" + risk_reason(original[2], params),
                "original_risk": original[2]["risk"],
                "selected_risk": recovered_score["risk"],
            }

    intervened = selected[0] != "policy"
    reason_prefix = "critical_" if variant in ("v5_critical_override", *V6_VARIANTS, *V7_VARIANTS, *V8_VARIANTS, *V9_VARIANTS, *V10_VARIANTS) and is_critical else ""
    reason = risk_reason(original[2], params, prefix=reason_prefix)
    return selected[1], {
        "intervened": intervened,
        "warning": not intervened,
        "candidate": selected[0],
        "reason": reason,
        "original_risk": original[2]["risk"],
        "selected_risk": selected[2]["risk"],
    }


def evaluate_single_trajectory_with_shield(model, dataset, cfg, traj_index, shield_params):
    traj = dataset[traj_index]
    single_dataset = SingleTrajDataset(traj, dataset.expert_mean, dataset.expert_std)
    env = MergingEnv(single_dataset)
    env.collision_margin = 1.0

    obs, _ = env.reset(seed=cfg.SEED + traj_index)
    terminated = False
    truncated = False
    max_steps = len(env.current_traj["ego_pos"]) + 50

    ep_reward = 0.0
    dense_ep_reward = 0.0
    speed_trace = []
    speed_y_mps_trace = []
    abs_acc_trace = []
    jerk_trace_mps3 = []
    ttc_trace = []
    thw_trace = []

    first_collision = None
    min_ttc_info = None
    min_thw_info = None
    last_info = {}
    step_count = 0
    shield_interventions = 0
    shield_warnings = 0
    consecutive_interventions = 0
    safe_steps = 0
    first_shield = None
    first_warning = None
    first_merge_step = None
    shield_candidate_counts = Counter()

    while not (terminated or truncated) and step_count < max_steps:
        policy_action, _ = model.predict(obs, deterministic=True)
        policy_action = np.asarray(policy_action, dtype=np.float32).reshape(-1)
        action, shield_info = shield_action(
            env=env,
            policy_action=policy_action,
            params=shield_params,
            step_count=step_count,
            shield_interventions=shield_interventions,
            consecutive_interventions=consecutive_interventions,
            safe_steps=safe_steps,
        )

        if shield_info["intervened"]:
            shield_interventions += 1
            consecutive_interventions += 1
            shield_candidate_counts[shield_info["candidate"]] += 1
            if first_shield is None:
                first_shield = {
                    "step": step_count,
                    "time_s": step_count * cfg.DT,
                    "reason": shield_info["reason"],
                    "candidate": shield_info["candidate"],
                    "original_action_x": float(policy_action[0]),
                    "original_action_y": float(policy_action[1]),
                    "action_x": float(action[0]),
                    "action_y": float(action[1]),
                    "original_risk": shield_info["original_risk"],
                    "selected_risk": shield_info["selected_risk"],
                }
        else:
            consecutive_interventions = 0

        if shield_info.get("warning", False):
            shield_warnings += 1
            if first_warning is None:
                first_warning = {
                    "step": step_count,
                    "time_s": step_count * cfg.DT,
                    "reason": shield_info["reason"],
                }

        obs, reward, terminated, truncated, info = env.step(action)
        last_info = dict(info)
        if first_merge_step is None and bool(info.get("is_merge_success", False)):
            first_merge_step = step_count

        px, py, vx, vy = [float(x) for x in env.ego_state]
        speed = float(np.hypot(vx, vy))
        ax_phys = float(action[0] * cfg.PHYS_STEER_MAX)
        ay_phys = float(action[1] * cfg.PHYS_ACC_MAX)
        min_ttc = float(info.get("eval_min_ttc", 20.0))
        min_thw = float(info.get("eval_min_thw", 10.0))
        nearest = nearest_vehicle_snapshot(env, px, py, vy)
        collided_now = bool(info.get("is_collided", False))
        if (
            not collided_now
            and min_ttc >= shield_params.get("lead_ttc_min", 3.0)
            and min_thw >= shield_params.get("follow_thw_min", 0.8)
        ):
            safe_steps += 1
        else:
            safe_steps = 0

        ep_reward += float(reward)
        dense_ep_reward += float(info.get("eval_dense_reward", 0.0))
        speed_trace.append(speed)
        speed_y_mps_trace.append(float(info.get("eval_vy_mps", vy * 0.3048)))
        abs_acc_trace.append(abs(ay_phys))
        jerk_trace_mps3.append(float(info.get("eval_abs_jerk_mps3", 0.0)))
        ttc_trace.append(min_ttc)
        thw_trace.append(min_thw)

        if min_ttc_info is None or min_ttc < min_ttc_info["min_ttc"]:
            min_ttc_info = {
                "min_ttc": min_ttc,
                "step": step_count,
                "time_s": step_count * cfg.DT,
                "px": px,
                "py": py,
                "action_x": float(action[0]),
                "action_y": float(action[1]),
                "nearest": nearest,
            }
        if min_thw_info is None or min_thw < min_thw_info["min_thw"]:
            min_thw_info = {
                "min_thw": min_thw,
                "step": step_count,
                "time_s": step_count * cfg.DT,
            }

        if first_collision is None and collided_now:
            first_collision = {
                "step": step_count,
                "time_s": step_count * cfg.DT,
                "px": px,
                "py": py,
                "speed": speed,
                "action_x": float(action[0]),
                "action_y": float(action[1]),
                "ax": ax_phys,
                "ay": ay_phys,
                "nearest": nearest,
            }

        step_count += 1

    px, py, vx, vy = [float(x) for x in env.ego_state]
    goal_xy = env.current_traj["ego_pos"][-1]
    dist_to_goal = float(np.linalg.norm(env.ego_state[:2] - goal_xy))
    lane_divider_x = cfg.X_MIN + cfg.LANE_WIDTH
    in_aux_lane = px > lane_divider_x
    in_target_lane = px < (lane_divider_x - 3.28)
    final_lane = "target" if in_target_lane else ("aux" if in_aux_lane else "transition")

    collided = bool(getattr(env, "has_collided_this_episode", False))
    endpoint_success = bool(last_info.get("is_endpoint_success", False))
    safety_success = bool(last_info.get("is_safety_success", False))
    merge_success = bool(last_info.get("is_merge_success", False))
    p_score = paper_score(speed_y_mps_trace, jerk_trace_mps3, ttc_trace, endpoint_success, collided)

    collision_nearest = first_collision["nearest"] if first_collision else {}
    min_ttc_nearest = min_ttc_info["nearest"] if min_ttc_info else {}
    shield_rate = float(shield_interventions / max(step_count, 1))
    warning_rate = float(shield_warnings / max(step_count, 1))
    most_common_candidate = shield_candidate_counts.most_common(1)[0][0] if shield_candidate_counts else ""
    hold_lane_count = sum(
        count for name, count in shield_candidate_counts.items() if "hold_lane" in name
    )
    delay_lateral_count = sum(
        count
        for name, count in shield_candidate_counts.items()
        if "delay_lateral" in name or "delay_merge" in name
    )
    hard_brake_hold_count = int(shield_candidate_counts.get("hard_brake_hold", 0))

    row = {
        "traj_index": traj_index,
        "filename": traj.get("filename", f"trajectory_{traj_index}.csv"),
        "steps": step_count,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "merge_success": merge_success,
        "endpoint_success": endpoint_success,
        "safety_success": safety_success,
        "collision": collided,
        "failure_type": classify_failure(endpoint_success, safety_success, collided, truncated),
        "episode_reward": float(ep_reward),
        "eval_dense_return": float(dense_ep_reward),
        "paper_score": p_score,
        "mean_speed_ftps": float(np.mean(speed_trace)) if speed_trace else 0.0,
        "mean_speed_mps": float(np.mean(speed_trace) * 0.3048) if speed_trace else 0.0,
        "mean_abs_acc_ftps2": float(np.mean(abs_acc_trace)) if abs_acc_trace else 0.0,
        "mean_abs_jerk_mps3": float(np.mean(np.abs(jerk_trace_mps3))) if jerk_trace_mps3 else 0.0,
        "min_ttc": float(np.min(ttc_trace)) if ttc_trace else 20.0,
        "min_ttc_step": min_ttc_info["step"] if min_ttc_info else "",
        "min_ttc_time_s": min_ttc_info["time_s"] if min_ttc_info else "",
        "min_thw": float(np.min(thw_trace)) if thw_trace else 10.0,
        "min_thw_step": min_thw_info["step"] if min_thw_info else "",
        "min_thw_time_s": min_thw_info["time_s"] if min_thw_info else "",
        "first_collision_step": first_collision["step"] if first_collision else "",
        "first_collision_time_s": first_collision["time_s"] if first_collision else "",
        "collision_px": first_collision["px"] if first_collision else "",
        "collision_py": first_collision["py"] if first_collision else "",
        "collision_speed_ftps": first_collision["speed"] if first_collision else "",
        "collision_action_x": first_collision["action_x"] if first_collision else "",
        "collision_action_y": first_collision["action_y"] if first_collision else "",
        "collision_ax_ftps2": first_collision["ax"] if first_collision else "",
        "collision_ay_ftps2": first_collision["ay"] if first_collision else "",
        "collision_nearest_vehicle": collision_nearest.get("vehicle", ""),
        "collision_nearest_dx": collision_nearest.get("dx", ""),
        "collision_nearest_dy": collision_nearest.get("dy", ""),
        "collision_nearest_rel_vy": collision_nearest.get("rel_vy", ""),
        "min_ttc_px": min_ttc_info["px"] if min_ttc_info else "",
        "min_ttc_py": min_ttc_info["py"] if min_ttc_info else "",
        "min_ttc_action_x": min_ttc_info["action_x"] if min_ttc_info else "",
        "min_ttc_action_y": min_ttc_info["action_y"] if min_ttc_info else "",
        "min_ttc_nearest_vehicle": min_ttc_nearest.get("vehicle", ""),
        "min_ttc_nearest_dx": min_ttc_nearest.get("dx", ""),
        "min_ttc_nearest_dy": min_ttc_nearest.get("dy", ""),
        "min_ttc_nearest_rel_vy": min_ttc_nearest.get("rel_vy", ""),
        "final_px": px,
        "final_py": py,
        "final_vx": vx,
        "final_vy": vy,
        "dist_to_goal": dist_to_goal,
        "in_target_lane_final": bool(in_target_lane),
        "in_aux_lane_final": bool(in_aux_lane),
        "expert_len": len(env.current_traj["ego_pos"]),
        "shield_enabled": True,
        "shield_variant": shield_params.get("shield_variant", "default"),
        "shield_interventions": shield_interventions,
        "shield_intervention_rate": shield_rate,
        "shield_warnings": shield_warnings,
        "shield_warning_rate": warning_rate,
        "most_common_shield_candidate": most_common_candidate,
        "hold_lane_count": hold_lane_count,
        "delay_lateral_count": delay_lateral_count,
        "hard_brake_hold_count": hard_brake_hold_count,
        "first_merge_step": first_merge_step if first_merge_step is not None else "",
        "first_merge_time_s": first_merge_step * cfg.DT if first_merge_step is not None else "",
        "final_lane": final_lane,
        "first_shield_step": first_shield["step"] if first_shield else "",
        "first_shield_time_s": first_shield["time_s"] if first_shield else "",
        "first_shield_reason": first_shield["reason"] if first_shield else "",
        "first_shield_candidate": first_shield["candidate"] if first_shield else "",
        "first_shield_original_action_x": first_shield["original_action_x"] if first_shield else "",
        "first_shield_original_action_y": first_shield["original_action_y"] if first_shield else "",
        "first_shield_action_x": first_shield["action_x"] if first_shield else "",
        "first_shield_action_y": first_shield["action_y"] if first_shield else "",
        "first_shield_original_risk": first_shield["original_risk"] if first_shield else "",
        "first_shield_selected_risk": first_shield["selected_risk"] if first_shield else "",
        "first_shield_warning_step": first_warning["step"] if first_warning else "",
        "first_shield_warning_time_s": first_warning["time_s"] if first_warning else "",
        "first_shield_warning_reason": first_warning["reason"] if first_warning else "",
    }
    return row


def summarize_with_shield(model_info, rows, split_name, thresholds):
    summary = summarize_rows(model_info, rows)
    annotated = annotate_summary(summary, split_name, thresholds)
    total_steps = sum(int(row["steps"]) for row in rows)
    interventions = sum(int(row["shield_interventions"]) for row in rows)
    warnings = sum(int(row["shield_warnings"]) for row in rows)
    episodes_with_intervention = sum(1 for row in rows if int(row["shield_interventions"]) > 0)
    episodes_with_warning = sum(1 for row in rows if int(row["shield_warnings"]) > 0)
    annotated["shield_interventions_total"] = interventions
    annotated["shield_intervention_rate"] = float(interventions / max(total_steps, 1))
    annotated["episodes_with_shield_intervention"] = episodes_with_intervention
    annotated["shield_warnings_total"] = warnings
    annotated["shield_warning_rate"] = float(warnings / max(total_steps, 1))
    annotated["episodes_with_shield_warning"] = episodes_with_warning
    return {key: annotated.get(key, "") for key in SHIELD_SUMMARY_FIELDS}


def summarize_merge_false(model_info, rows, split_name):
    total = len(rows)
    merge_false = [row for row in rows if not bool(row.get("merge_success", False))]
    endpoint_true_merge_false = [
        row for row in merge_false if bool(row.get("endpoint_success", False))
    ]
    candidate_counter = Counter(
        row.get("most_common_shield_candidate", "")
        for row in merge_false
        if row.get("most_common_shield_candidate", "")
    )
    first_candidate_counter = Counter(
        row.get("first_shield_candidate", "")
        for row in merge_false
        if row.get("first_shield_candidate", "")
    )
    final_lane_counter = Counter(row.get("final_lane", "") for row in merge_false)

    def mean_float(field):
        values = []
        for row in merge_false:
            value = row.get(field, "")
            if value == "":
                continue
            values.append(float(value))
        return float(np.mean(values)) if values else 0.0

    return {
        "model_tag": model_info["tag"],
        "split": split_name,
        "total": total,
        "merge_false_count": len(merge_false),
        "merge_false_rate": float(len(merge_false) / max(total, 1)),
        "endpoint_true_merge_false_count": len(endpoint_true_merge_false),
        "endpoint_true_merge_false_rate": float(len(endpoint_true_merge_false) / max(total, 1)),
        "endpoint_false_merge_false_count": sum(
            1 for row in merge_false if not bool(row.get("endpoint_success", False))
        ),
        "safety_false_merge_false_count": sum(
            1 for row in merge_false if not bool(row.get("safety_success", False))
        ),
        "collision_merge_false_count": sum(
            1 for row in merge_false if bool(row.get("collision", False))
        ),
        "mean_shield_intervention_rate": mean_float("shield_intervention_rate"),
        "mean_dist_to_goal": mean_float("dist_to_goal"),
        "final_target_count": final_lane_counter.get("target", 0),
        "final_transition_count": final_lane_counter.get("transition", 0),
        "final_aux_count": final_lane_counter.get("aux", 0),
        "most_common_shield_candidate": (
            candidate_counter.most_common(1)[0][0] if candidate_counter else ""
        ),
        "most_common_first_shield_candidate": (
            first_candidate_counter.most_common(1)[0][0] if first_candidate_counter else ""
        ),
        "mean_hold_lane_count": mean_float("hold_lane_count"),
        "mean_delay_lateral_count": mean_float("delay_lateral_count"),
        "mean_hard_brake_hold_count": mean_float("hard_brake_hold_count"),
        "endpoint_true_merge_false_filenames": ";".join(
            row.get("filename", "") for row in endpoint_true_merge_false
        ),
    }


def write_report(output_dir, summaries, hard_filenames, shield_params):
    lines = [
        "# Safety shield evaluation",
        "",
        f"Created at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "This is evaluation-only. It does not change training, rewards, environment logic, or checkpoint saving.",
        "",
        "## Shield parameters",
        "",
        "```json",
        json.dumps(shield_params, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Hard-case list",
        "",
        f"N={len(hard_filenames)}",
        "",
        "```text",
        *hard_filenames,
        "```",
        "",
        "## Summary",
        "",
        "| Model | Split | N | Endpoint | Safety | Collision | Collisions | Mean TTC | Mean THW | Shield Rate | Warning Rate | Episodes Shielded | Pass |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summaries:
        lines.append(
            "| {model} | {split} | {total} | {endpoint:.3f} | {safety:.3f} | "
            "{collision:.3f} | {collision_count} | {ttc:.3f} | {thw:.3f} | "
            "{shield_rate:.3f} | {warning_rate:.3f} | {episodes} | {passed} |".format(
                model=row["model_tag"],
                split=row["split"],
                total=row["total"],
                endpoint=float(row["endpoint_success_rate"]),
                safety=float(row["safety_success_rate"]),
                collision=float(row["collision_rate"]),
                collision_count=row["collision_count"],
                ttc=float(row["mean_min_ttc"]),
                thw=float(row["mean_min_thw"]),
                shield_rate=float(row["shield_intervention_rate"]),
                warning_rate=float(row["shield_warning_rate"]),
                episodes=row["episodes_with_shield_intervention"],
                passed=row["protocol_pass"],
            )
        )
    lines.extend([
        "",
        "## Interpretation guide",
        "",
        "- If hard15 collision drops while endpoint stays high, the issue is likely target-lane gap safety rather than the AIRL policy as a whole.",
        "- If collision drops but endpoint collapses, the shield is too conservative or candidate actions are too braking-heavy.",
        "- If collision does not drop, the first shield is not looking far enough ahead or its candidate action set is insufficient.",
        "",
    ])
    (output_dir / "shield_report.md").write_text("\n".join(lines), encoding="utf-8-sig")


def main():
    args = parse_args()
    models = get_models(args)
    hard_list_path, hard_filenames = load_hard_case_filenames(args.hard_list)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else ROOT_DIR / "train_log" / f"safety_shield_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    variant = args.shield_variant
    margin_variants = ("v3_gate_margin", "v4_combo", "v5_critical_override", *V6_VARIANTS, *V7_VARIANTS, *V8_VARIANTS, *V9_VARIANTS, *V10_VARIANTS)
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
    recovery_enabled = variant in ("v5_critical_override", "v6b_recovery_merge", "v6c_margin020", *V7_VARIANTS, *V8_VARIANTS, *V9_VARIANTS, *V10_VARIANTS)
    recovery_safe_steps = 3 if variant in ("v6b_recovery_merge", "v6c_margin020") else 5
    recovery_policy_steps = 6 if variant in ("v6b_recovery_merge", "v6c_margin020") else 8
    shield_params = {
        "shield_variant": variant,
        "prediction_horizon": args.prediction_horizon,
        "lead_ttc_min": args.lead_ttc_min,
        "follow_ttc_min": args.follow_ttc_min,
        "lead_thw_min": args.lead_thw_min,
        "follow_thw_min": args.follow_thw_min,
        "risk_improvement_margin": risk_improvement_margin,
        "max_intervention_rate": max_intervention_rate,
        "max_consecutive_interventions": 10 if variant in ("v4_combo", "v5_critical_override", *V6_VARIANTS, *V7_VARIANTS, *V8_VARIANTS, *V9_VARIANTS, *V10_VARIANTS) else 10 ** 9,
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
    thresholds = {
        "endpoint": args.endpoint_threshold,
        "safety": args.safety_threshold,
        "collision": args.collision_threshold,
    }

    print("=" * 80)
    print("Evaluation-only safety shield")
    print(f"Output: {output_dir}")
    print(f"Hard-case list: {hard_list_path}")
    print(f"Hard cases: {len(hard_filenames)}")
    print(f"Include full split: {args.include_full}")
    print(f"Shield params: {shield_params}")
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
    splits = [("hard15", hard_indices)]
    if args.include_full:
        splits.insert(0, ("full", list(range(len(dataset)))))

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "dataset_size": len(dataset),
        "hard_case_count": len(hard_indices),
        "hard_case_list": str(hard_list_path),
        "include_full": args.include_full,
        "thresholds": thresholds,
        "shield_params": shield_params,
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
    write_csv(
        output_dir / "hard_case_list_used.csv",
        [{"filename": name} for name in hard_filenames],
        ["filename"],
    )

    summaries = []
    merge_false_summaries = []
    rows_by_split = {split_name: {} for split_name, _ in splits}
    for model_info in models:
        tag = model_info["tag"]
        print("\n" + "-" * 80)
        print(f"Loading {tag}")
        model = PPO.load(str(model_info["checkpoint"]), device=device)

        for split_name, indices in splits:
            rows = []
            for traj_index in tqdm(indices, desc=f"{tag}:{split_name}:shield"):
                row = evaluate_single_trajectory_with_shield(
                    model=model,
                    dataset=dataset,
                    cfg=cfg,
                    traj_index=traj_index,
                    shield_params=shield_params,
                )
                row["model_tag"] = tag
                rows.append(row)

            prefix = f"{tag}_{split_name}_shield"
            write_csv(output_dir / f"{prefix}_trajectory_details.csv", rows, SHIELD_DETAIL_FIELDS)
            write_csv(
                output_dir / f"{prefix}_failure_cases.csv",
                [row for row in rows if row["failure_type"] != "success"],
                SHIELD_DETAIL_FIELDS,
            )
            write_csv(
                output_dir / f"{prefix}_collision_cases.csv",
                [row for row in rows if row["collision"]],
                SHIELD_DETAIL_FIELDS,
            )
            write_csv(
                output_dir / f"{prefix}_merge_false_cases.csv",
                [row for row in rows if not row["merge_success"]],
                SHIELD_DETAIL_FIELDS,
            )

            rows_by_split[split_name][tag] = rows
            summary = summarize_with_shield(model_info, rows, split_name, thresholds)
            summaries.append(summary)
            merge_false_summaries.append(summarize_merge_false(model_info, rows, split_name))

            print(
                f"{tag}/{split_name}+shield: N={summary['total']}, "
                f"endpoint={float(summary['endpoint_success_rate']):.3f}, "
                f"safety={float(summary['safety_success_rate']):.3f}, "
                f"collision={float(summary['collision_rate']):.3f}, "
                f"collisions={summary['collision_count']}, "
                f"shield_rate={float(summary['shield_intervention_rate']):.3f}, "
                f"pass={summary['protocol_pass']}"
            )

    write_csv(output_dir / "shield_protocol_summary.csv", summaries, SHIELD_SUMMARY_FIELDS)
    write_csv(
        output_dir / "merge_false_summary.csv",
        merge_false_summaries,
        MERGE_FALSE_SUMMARY_FIELDS,
    )
    for split_name, by_model in rows_by_split.items():
        if by_model:
            write_overlap(output_dir / f"{split_name}_shield_overlap", by_model)
    write_report(output_dir, summaries, hard_filenames, shield_params)

    print("\n" + "=" * 80)
    print(f"Saved safety shield evaluation to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
