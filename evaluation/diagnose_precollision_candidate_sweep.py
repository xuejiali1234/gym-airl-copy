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
from model.attention_net import AttentionFeaturesExtractor, GoalConditionedMLPFeaturesExtractor  # noqa: F401
from model.predictive_safety_oracle import PredictiveSafetyOracle
from utils.data_loader import MergingDataset


DEFAULT_HARD_CASE_DIR = ROOT_DIR / "train_log" / "failure_case_full_eval_20260427_205244"
DEFAULT_HARD_CASE_LIST = DEFAULT_HARD_CASE_DIR / "common15_trajectory_summary_for_gpt.csv"
OFFSETS_SECONDS = [1.5, 1.0, 0.7, 0.5, 0.3, 0.0]

TARGET_MODELS = [
    {
        "tag": "True_Decay250_265",
        "checkpoint": ROOT_DIR
        / "train_log"
        / "baseline_attn_goal_safe_branch_aux_probe_P300_U220_L5e6_D230_Decay250_20260427_165403"
        / "checkpoints"
        / "baseline_policy_attn_goal_safe_branch_aux_probe_P300_U220_L5e6_D230_Decay250_best_epoch_265.zip",
        "note": "True historical Decay250 best epoch 265.",
    },
    {
        "tag": "True_Decay250_270",
        "checkpoint": ROOT_DIR
        / "train_log"
        / "baseline_attn_goal_safe_branch_aux_probe_P300_U220_L5e6_D230_Decay250_20260427_165403"
        / "checkpoints"
        / "baseline_policy_attn_goal_safe_branch_aux_probe_P300_U220_L5e6_D230_Decay250_best_epoch_270.zip",
        "note": "True historical Decay250 checkpoint 270.",
    },
    {
        "tag": "P30_CPairD250_292",
        "checkpoint": ROOT_DIR
        / "train_log"
        / "baseline_attn_goal_safe_branch_aux_probe_P30_CPairD250_NoLateLR_Save1_20260502_215110"
        / "checkpoints"
        / "baseline_policy_attn_goal_safe_branch_aux_probe_P30_CPairD250_NoLateLR_Save1_epoch_292.zip",
        "note": "P30 balance candidate.",
    },
    {
        "tag": "P30_CPairD250_298",
        "checkpoint": ROOT_DIR
        / "train_log"
        / "baseline_attn_goal_safe_branch_aux_probe_P30_CPairD250_NoLateLR_Save1_20260502_215110"
        / "checkpoints"
        / "baseline_policy_attn_goal_safe_branch_aux_probe_P30_CPairD250_NoLateLR_Save1_epoch_298.zip",
        "note": "P30 safety-first candidate.",
    },
]


DETAIL_FIELDS = [
    "model_tag",
    "split",
    "checkpoint",
    "traj_index",
    "filename",
    "collision_step",
    "collision_time_s",
    "collision_object",
    "offset_s",
    "eval_step",
    "eval_time_s",
    "history_truncated",
    "candidate_set",
    "policy_action_x",
    "policy_action_y",
    "policy_risk_clipped",
    "policy_risk_raw",
    "min_candidate_action_type",
    "min_candidate_risk_clipped",
    "min_candidate_risk_raw",
    "min_candidate_action_x",
    "min_candidate_action_y",
    "risk_gap_clipped",
    "risk_gap_raw",
    "task_safe_exists",
    "task_safe_action_type",
    "task_safe_action_category",
    "task_safe_action_x",
    "task_safe_action_y",
    "task_safe_risk_clipped",
    "task_safe_risk_raw",
    "all_candidates_saturated",
]

SUMMARY_FIELDS = [
    "model_tag",
    "split",
    "candidate_set",
    "offset_s",
    "samples",
    "lead_collision_count",
    "follow_collision_count",
    "side_collision_count",
    "unknown_collision_count",
    "policy_risk_clipped_mean",
    "min_candidate_risk_clipped_mean",
    "policy_risk_raw_mean",
    "min_candidate_risk_raw_mean",
    "risk_gap_clipped_mean",
    "risk_gap_raw_mean",
    "task_safe_exists_rate",
    "all_candidates_saturated_rate",
    "safe_is_hold_rate",
    "safe_is_decel_rate",
    "safe_is_delay_rate",
    "safe_is_merge_rate",
    "safe_is_speedmatch_rate",
    "safe_is_accel_rate",
    "safe_is_policy_rate",
    "most_common_task_safe_type",
]


class SingleTrajDataset:
    def __init__(self, traj, expert_mean, expert_std):
        self.trajectories = [traj]
        self.expert_mean = expert_mean
        self.expert_std = expert_std

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.trajectories[idx]


def parse_args():
    parser = argparse.ArgumentParser(description="D32 pre-collision candidate sweep.")
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="TAG=CHECKPOINT",
        help="Model to diagnose. Can be repeated. Defaults to the built-in P30/Decay250 set.",
    )
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated model tags/prefixes to run from the selected model list.",
    )
    parser.add_argument(
        "--hard-list",
        default=str(DEFAULT_HARD_CASE_LIST),
        help="CSV containing a filename column for hard cases.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory. Defaults to train_log/D32_PreCollision_CandidateSweep_<timestamp>.",
    )
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
    return {
        "tag": tag.strip(),
        "checkpoint": Path(checkpoint.strip().strip('"')).expanduser().resolve(),
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


def load_dataset(device):
    data_paths = [
        ROOT_DIR / "data" / "lane_change_trajectories-0750am-0805am",
        ROOT_DIR / "data" / "lane_change_trajectories-0805am-0820am",
        ROOT_DIR / "data" / "lane_change_trajectories-0820am-0835am",
    ]
    return MergingDataset([str(p) for p in data_paths], device=device)


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
        raise ValueError("Hard-case filenames missing from dataset: " + ", ".join(sorted(missing)))
    return matched


def nearest_vehicle_snapshot(env, px, py, vy):
    surr = env._get_surround_at_t(env.t)
    names = ["L6_lead", "L5_lead", "L5_follow", "L6_follow"]
    best = {"vehicle": "", "dx": np.nan, "dy": np.nan, "rel_vy": np.nan}
    best_dist = float("inf")
    for vehicle_idx, name in enumerate(names):
        base = vehicle_idx * 4
        ox = float(surr[base])
        oy = float(surr[base + 1])
        ovy = float(surr[base + 2])
        if ox == 0.0 and oy == 0.0:
            continue
        dx = ox - px
        dy = oy - py
        dist = float(np.hypot(dx, dy))
        if dist < best_dist:
            best_dist = dist
            best = {
                "vehicle": name,
                "dx": float(dx),
                "dy": float(dy),
                "rel_vy": float(ovy - vy),
            }
    return best


def classify_collision_object(nearest_vehicle, dx, dy):
    if nearest_vehicle in {"L6_lead", "L5_lead"}:
        return "lead"
    if nearest_vehicle in {"L5_follow", "L6_follow"}:
        return "follow"
    if dx is not None and dy is not None and not (np.isnan(dx) or np.isnan(dy)):
        if abs(dx) > 3.0 and abs(dy) < 18.0:
            return "side"
    return "unknown"


def build_candidate_sets(action_tensor):
    lateral = action_tensor[:, 0]
    longitudinal = action_tensor[:, 1]

    hold = torch.stack([torch.zeros_like(lateral), longitudinal], dim=-1)
    delay = torch.stack([0.5 * lateral.clamp(max=0.0), longitudinal], dim=-1)
    mild_decel = torch.stack([lateral, (longitudinal - 0.25).clamp(-1.0, 1.0)], dim=-1)
    mild_accel = torch.stack([lateral, (longitudinal + 0.25).clamp(-1.0, 1.0)], dim=-1)
    mild_merge = torch.stack([(lateral - 0.25).clamp(-1.0, 1.0), longitudinal], dim=-1)

    hold_decel = torch.stack([torch.zeros_like(lateral), (longitudinal - 0.25).clamp(-1.0, 1.0)], dim=-1)
    strong_decel = torch.stack([lateral, (longitudinal - 0.5).clamp(-1.0, 1.0)], dim=-1)
    delay_decel = torch.stack([0.5 * lateral.clamp(max=0.0), (longitudinal - 0.25).clamp(-1.0, 1.0)], dim=-1)
    speedmatch = torch.stack([lateral, (longitudinal - 0.12).clamp(-1.0, 1.0)], dim=-1)
    delay_speedmatch = torch.stack([0.5 * lateral.clamp(max=0.0), (longitudinal - 0.12).clamp(-1.0, 1.0)], dim=-1)
    mild_merge_decel = torch.stack([(lateral - 0.25).clamp(-1.0, 1.0), (longitudinal - 0.25).clamp(-1.0, 1.0)], dim=-1)
    mild_merge_speedmatch = torch.stack([(lateral - 0.25).clamp(-1.0, 1.0), (longitudinal - 0.12).clamp(-1.0, 1.0)], dim=-1)
    strong_merge = torch.stack([(lateral - 0.5).clamp(-1.0, 1.0), longitudinal], dim=-1)
    strong_merge_decel = torch.stack([(lateral - 0.5).clamp(-1.0, 1.0), (longitudinal - 0.25).clamp(-1.0, 1.0)], dim=-1)
    strong_merge_accel = torch.stack([(lateral - 0.5).clamp(-1.0, 1.0), (longitudinal + 0.25).clamp(-1.0, 1.0)], dim=-1)

    base = {
        "policy": action_tensor,
        "hold": hold,
        "delay": delay,
        "decel": mild_decel,
        "accel": mild_accel,
        "merge": mild_merge,
    }
    cand_v2 = {
        **base,
        "hold_decel": hold_decel,
        "delay_decel": delay_decel,
        "speedmatch": speedmatch,
        "merge_decel": mild_merge_decel,
    }
    cand_v2_leadaware = {
        **cand_v2,
        "strong_decel": strong_decel,
        "delay_speedmatch": delay_speedmatch,
        "merge_speedmatch": mild_merge_speedmatch,
        "strong_merge": strong_merge,
        "strong_merge_decel": strong_merge_decel,
        "strong_merge_accel": strong_merge_accel,
    }
    return {
        "current": base,
        "candidate_v2": cand_v2,
        "candidate_v2_leadaware": cand_v2_leadaware,
    }


def action_category(name):
    if name == "policy":
        return "policy"
    if "speedmatch" in name:
        return "speedmatch"
    if "delay" in name:
        return "delay"
    if "merge" in name:
        return "merge"
    if "decel" in name:
        return "decel"
    if "hold" in name:
        return "hold"
    if "accel" in name:
        return "accel"
    return "other"


def task_progress_bonus(name):
    category = action_category(name)
    if category == "merge":
        return 0.035
    if category == "delay":
        return 0.025
    if category == "speedmatch":
        return 0.015
    if category == "policy":
        return 0.015
    if category == "decel":
        return 0.010
    return 0.0


def task_penalty(name):
    penalty = 0.0
    if name == "hold":
        penalty += 0.020
    if name == "strong_decel":
        penalty += 0.040
    if name == "strong_merge_accel":
        penalty += 0.030
    if name == "accel":
        penalty += 0.015
    return penalty


def analyze_candidate_sets(oracle, obs_before, action):
    state_tensor = torch.as_tensor(obs_before[None, :], dtype=torch.float32)
    action_tensor = torch.as_tensor(action[None, :], dtype=torch.float32)
    output_rows = []

    for candidate_set_name, candidates_by_name in build_candidate_sets(action_tensor).items():
        candidate_names = list(candidates_by_name.keys())
        candidate_tensor = torch.stack([candidates_by_name[name] for name in candidate_names], dim=1).clamp(-1.0, 1.0)
        num_candidates = candidate_tensor.shape[1]
        state_rep = state_tensor.repeat(num_candidates, 1)
        flat_actions = candidate_tensor.reshape(-1, action_tensor.shape[-1])
        analysis = oracle.analyze_batch(state_rep, flat_actions)
        risks_clipped = analysis["risk_score"].reshape(-1).cpu().numpy()
        risks_raw = analysis["risk_score_unclipped"].reshape(-1).cpu().numpy()
        actions_np = candidate_tensor[0].cpu().numpy()

        policy_idx = candidate_names.index("policy")
        policy_risk_clipped = float(risks_clipped[policy_idx])
        policy_risk_raw = float(risks_raw[policy_idx])
        min_idx = int(np.argmin(risks_raw))

        candidate_task_scores = []
        for idx, name in enumerate(candidate_names):
            task_score = float(risks_raw[idx]) - task_progress_bonus(name) + task_penalty(name)
            candidate_task_scores.append(task_score)

        eligible_task_indices = [
            idx
            for idx, name in enumerate(candidate_names)
            if idx != policy_idx and float(risks_raw[idx]) + 1e-6 < policy_risk_raw
        ]
        if eligible_task_indices:
            task_safe_idx = min(eligible_task_indices, key=lambda idx: candidate_task_scores[idx])
            task_safe_exists = True
            task_safe_type = candidate_names[task_safe_idx]
            task_safe_action = actions_np[task_safe_idx]
            task_safe_risk_clipped = float(risks_clipped[task_safe_idx])
            task_safe_risk_raw = float(risks_raw[task_safe_idx])
        else:
            task_safe_idx = None
            task_safe_exists = False
            task_safe_type = ""
            task_safe_action = np.array([np.nan, np.nan], dtype=np.float32)
            task_safe_risk_clipped = np.nan
            task_safe_risk_raw = np.nan

        all_candidates_saturated = bool(np.all(risks_clipped >= 0.999))
        min_action = actions_np[min_idx]
        output_rows.append(
            {
                "candidate_set": candidate_set_name,
                "policy_risk_clipped": policy_risk_clipped,
                "policy_risk_raw": policy_risk_raw,
                "min_candidate_action_type": candidate_names[min_idx],
                "min_candidate_risk_clipped": float(risks_clipped[min_idx]),
                "min_candidate_risk_raw": float(risks_raw[min_idx]),
                "risk_gap_clipped": float(policy_risk_clipped - float(risks_clipped[min_idx])),
                "risk_gap_raw": float(policy_risk_raw - float(risks_raw[min_idx])),
                "task_safe_exists": task_safe_exists,
                "task_safe_action_type": task_safe_type,
                "task_safe_action_category": action_category(task_safe_type) if task_safe_type else "",
                "task_safe_action_x": float(task_safe_action[0]) if not np.isnan(task_safe_action[0]) else "",
                "task_safe_action_y": float(task_safe_action[1]) if not np.isnan(task_safe_action[1]) else "",
                "task_safe_risk_clipped": float(task_safe_risk_clipped) if not np.isnan(task_safe_risk_clipped) else "",
                "task_safe_risk_raw": float(task_safe_risk_raw) if not np.isnan(task_safe_risk_raw) else "",
                "all_candidates_saturated": all_candidates_saturated,
                "policy_action_x": float(actions_np[policy_idx][0]),
                "policy_action_y": float(actions_np[policy_idx][1]),
                "min_candidate_action_x": float(min_action[0]),
                "min_candidate_action_y": float(min_action[1]),
            }
        )
    return output_rows


def collect_trajectory_trace(model, dataset, cfg, oracle, traj_index):
    traj = dataset[traj_index]
    single_dataset = SingleTrajDataset(traj, dataset.expert_mean, dataset.expert_std)
    env = MergingEnv(single_dataset)
    env.collision_margin = 1.0

    obs, _ = env.reset(seed=cfg.SEED + traj_index)
    terminated = False
    truncated = False
    max_steps = len(env.current_traj["ego_pos"]) + 50

    step_records = []
    step_count = 0
    first_collision_step = None
    last_info = {}

    while not (terminated or truncated) and step_count < max_steps:
        obs_before = np.asarray(obs, dtype=np.float32).copy()
        pre_px, pre_py, pre_vx, pre_vy = [float(x) for x in env.ego_state]
        nearest_pre = nearest_vehicle_snapshot(env, pre_px, pre_py, pre_vy)

        action, _ = model.predict(obs_before, deterministic=True)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        analysis = oracle.analyze_batch(
            torch.as_tensor(obs_before[None, :], dtype=torch.float32),
            torch.as_tensor(action[None, :], dtype=torch.float32),
        )

        obs, reward, terminated, truncated, info = env.step(action)
        last_info = dict(info)

        collided_now = bool(info.get("is_collided", False))
        if first_collision_step is None and collided_now:
            first_collision_step = step_count

        step_records.append(
            {
                "step": step_count,
                "time_s": float(step_count * cfg.DT),
                "obs_before": obs_before,
                "action": action.copy(),
                "policy_risk_clipped": float(analysis["risk_score"][0, 0].item()),
                "policy_risk_raw": float(analysis["risk_score_unclipped"][0, 0].item()),
                "nearest_pre": nearest_pre,
                "collided_now": collided_now,
                "reward": float(reward),
            }
        )
        step_count += 1

    collided = bool(getattr(env, "has_collided_this_episode", False))
    filename = traj.get("filename", f"trajectory_{traj_index}.csv")
    endpoint_success = bool(last_info.get("is_endpoint_success", False))
    merge_success = bool(last_info.get("is_merge_success", False))
    safety_success = bool(last_info.get("is_safety_success", False))

    return {
        "collided": collided,
        "traj_index": traj_index,
        "filename": filename,
        "endpoint_success": endpoint_success,
        "merge_success": merge_success,
        "safety_success": safety_success,
        "collision_step": first_collision_step,
        "step_records": step_records,
    }


def diagnose_collision_trace(model_info, split_name, trace, cfg, oracle):
    if not trace["collided"] or trace["collision_step"] is None:
        return []

    collision_step = int(trace["collision_step"])
    collision_record = trace["step_records"][collision_step]
    nearest = collision_record["nearest_pre"]
    collision_object = classify_collision_object(
        nearest_vehicle=nearest.get("vehicle", ""),
        dx=nearest.get("dx", np.nan),
        dy=nearest.get("dy", np.nan),
    )

    rows = []
    for offset_s in OFFSETS_SECONDS:
        offset_steps = int(round(offset_s / cfg.DT))
        eval_step = max(0, collision_step - offset_steps)
        step_record = trace["step_records"][eval_step]
        history_truncated = eval_step != collision_step - offset_steps

        for candidate_row in analyze_candidate_sets(
            oracle=oracle,
            obs_before=step_record["obs_before"],
            action=step_record["action"],
        ):
            rows.append(
                {
                    "model_tag": model_info["tag"],
                    "split": split_name,
                    "checkpoint": str(model_info["checkpoint"]),
                    "traj_index": trace["traj_index"],
                    "filename": trace["filename"],
                    "collision_step": collision_step,
                    "collision_time_s": float(collision_step * cfg.DT),
                    "collision_object": collision_object,
                    "offset_s": float(offset_s),
                    "eval_step": eval_step,
                    "eval_time_s": float(step_record["time_s"]),
                    "history_truncated": history_truncated,
                    **candidate_row,
                }
            )
    return rows


def summarize_rows(rows):
    summaries = []
    group_keys = sorted({(row["model_tag"], row["split"], row["candidate_set"], row["offset_s"]) for row in rows})
    for model_tag, split_name, candidate_set, offset_s in group_keys:
        chunk = [
            row
            for row in rows
            if row["model_tag"] == model_tag
            and row["split"] == split_name
            and row["candidate_set"] == candidate_set
            and row["offset_s"] == offset_s
        ]
        if not chunk:
            continue

        action_counter = Counter(row["task_safe_action_type"] for row in chunk if row["task_safe_exists"] and row["task_safe_action_type"])
        safe_category_counter = Counter(row["task_safe_action_category"] for row in chunk if row["task_safe_exists"] and row["task_safe_action_category"])
        object_counter = Counter(row["collision_object"] for row in chunk)

        def mean_of(key):
            return float(np.mean([float(row[key]) for row in chunk]))

        def rate_of(predicate):
            return float(np.mean([1.0 if predicate(row) else 0.0 for row in chunk]))

        summaries.append(
            {
                "model_tag": model_tag,
                "split": split_name,
                "candidate_set": candidate_set,
                "offset_s": float(offset_s),
                "samples": len(chunk),
                "lead_collision_count": object_counter.get("lead", 0),
                "follow_collision_count": object_counter.get("follow", 0),
                "side_collision_count": object_counter.get("side", 0),
                "unknown_collision_count": object_counter.get("unknown", 0),
                "policy_risk_clipped_mean": mean_of("policy_risk_clipped"),
                "min_candidate_risk_clipped_mean": mean_of("min_candidate_risk_clipped"),
                "policy_risk_raw_mean": mean_of("policy_risk_raw"),
                "min_candidate_risk_raw_mean": mean_of("min_candidate_risk_raw"),
                "risk_gap_clipped_mean": mean_of("risk_gap_clipped"),
                "risk_gap_raw_mean": mean_of("risk_gap_raw"),
                "task_safe_exists_rate": rate_of(lambda row: row["task_safe_exists"]),
                "all_candidates_saturated_rate": rate_of(lambda row: row["all_candidates_saturated"]),
                "safe_is_hold_rate": safe_category_counter.get("hold", 0) / len(chunk),
                "safe_is_decel_rate": safe_category_counter.get("decel", 0) / len(chunk),
                "safe_is_delay_rate": safe_category_counter.get("delay", 0) / len(chunk),
                "safe_is_merge_rate": safe_category_counter.get("merge", 0) / len(chunk),
                "safe_is_speedmatch_rate": safe_category_counter.get("speedmatch", 0) / len(chunk),
                "safe_is_accel_rate": safe_category_counter.get("accel", 0) / len(chunk),
                "safe_is_policy_rate": safe_category_counter.get("policy", 0) / len(chunk),
                "most_common_task_safe_type": action_counter.most_common(1)[0][0] if action_counter else "",
            }
        )
    return summaries


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, metadata, summaries):
    lines = [
        "# D32 Pre-Collision Candidate Sweep",
        "",
        f"Created at: {metadata['created_at']}",
        "",
        "## Objects",
        "",
    ]
    for model_info in metadata["models"]:
        lines.append(f"- `{model_info['tag']}`: `{model_info['checkpoint']}`")
    lines.extend(
        [
            "",
            "## Offsets",
            "",
            "- `1.5s / 1.0s / 0.7s / 0.5s / 0.3s / 0.0s` before the first collision step.",
            "",
            "## Candidate Sets",
            "",
            "- `current`: current six-action candidate set.",
            "- `candidate_v2`: current set plus hold/decel and speed-match variants.",
            "- `candidate_v2_leadaware`: candidate_v2 plus stronger lead-avoid / speed-match / strong-merge variants.",
            "",
            "## Reading Guide",
            "",
            "- `policy_risk_clipped/raw` show the policy action severity at the diagnosis step.",
            "- `min_candidate_*` shows the best candidate under pure risk minimization.",
            "- `task_safe_*` shows whether there exists a lower-risk candidate that still scores well under a small task-compatibility heuristic.",
            "- `all_candidates_saturated_rate` close to `1.0` means the whole candidate set is effectively saturated at clipped risk `1.0`.",
            "",
            "## Summary",
            "",
        ]
    )

    for row in summaries:
        lines.extend(
            [
                f"### {row['model_tag']} / {row['split']} / {row['candidate_set']} / t-{row['offset_s']:.1f}s",
                "",
                f"- samples: {row['samples']}",
                f"- raw risk gap mean: {row['risk_gap_raw_mean']:.3f}",
                f"- clipped risk gap mean: {row['risk_gap_clipped_mean']:.3f}",
                f"- task-safe exists rate: {row['task_safe_exists_rate']:.3f}",
                f"- all candidates saturated rate: {row['all_candidates_saturated_rate']:.3f}",
                f"- most common task-safe type: `{row['most_common_task_safe_type']}`",
                "",
            ]
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    models = get_models(args)

    hard_list_path, hard_filenames = load_hard_case_filenames(args.hard_list)
    if args.dry_run:
        print("Selected models:")
        for item in models:
            print(f"  - {item['tag']}: {item['checkpoint']}")
        print("Hard15 filenames:")
        for filename in hard_filenames:
            print(f"  - {filename}")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else ROOT_DIR / "train_log" / f"D32_PreCollision_CandidateSweep_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config()
    device = torch.device("cpu")
    dataset = load_dataset(device=device)
    hard_indices = dataset_indices_by_filename(dataset, hard_filenames)
    full_indices = list(range(len(dataset.trajectories)))

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "hard_case_list": str(hard_list_path),
        "models": [
            {
                "tag": item["tag"],
                "checkpoint": str(item["checkpoint"]),
                "note": item.get("note", ""),
            }
            for item in models
        ],
        "offsets_seconds": OFFSETS_SECONDS,
    }

    detail_rows = []
    for model_info in models:
        print(f"[D32] Loading {model_info['tag']} -> {model_info['checkpoint']}")
        model = PPO.load(str(model_info["checkpoint"]), device="cpu")
        oracle = PredictiveSafetyOracle(cfg, dataset.expert_mean, dataset.expert_std)

        for split_name, indices in (("full", full_indices), ("hard15", hard_indices)):
            iterator = tqdm(indices, desc=f"{model_info['tag']} {split_name}", leave=False)
            for traj_index in iterator:
                trace = collect_trajectory_trace(model, dataset, cfg, oracle, traj_index)
                if not trace["collided"]:
                    continue
                detail_rows.extend(diagnose_collision_trace(model_info, split_name, trace, cfg, oracle))

    summary_rows = summarize_rows(detail_rows)

    write_csv(output_dir / "precollision_candidate_sweep.csv", detail_rows, DETAIL_FIELDS)
    write_csv(output_dir / "precollision_candidate_sweep_summary.csv", summary_rows, SUMMARY_FIELDS)
    write_report(output_dir / "precollision_candidate_sweep_report.md", metadata, summary_rows)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[D32] Wrote detail rows to: {output_dir / 'precollision_candidate_sweep.csv'}")
    print(f"[D32] Wrote summary rows to: {output_dir / 'precollision_candidate_sweep_summary.csv'}")
    print(f"[D32] Wrote report to: {output_dir / 'precollision_candidate_sweep_report.md'}")


if __name__ == "__main__":
    main()
