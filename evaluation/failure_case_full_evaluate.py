import csv
import json
import os
import sys
from collections import defaultdict
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
from utils.data_loader import MergingDataset

# Keep custom policy/reward classes importable when SB3 unpickles saved policies.
from model.attention_net import AttentionFeaturesExtractor, GoalConditionedMLPFeaturesExtractor  # noqa: F401


class SingleTrajDataset:
    def __init__(self, traj, expert_mean, expert_std):
        self.trajectories = [traj]
        self.expert_mean = expert_mean
        self.expert_std = expert_std

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.trajectories[idx]


TARGET_MODELS = [
    {
        "tag": "Decay250_epoch265",
        "checkpoint": ROOT_DIR
        / "train_log"
        / "baseline_attn_goal_safe_branch_aux_probe_P300_U220_L5e6_D230_Decay250_20260427_165403"
        / "checkpoints"
        / "baseline_policy_attn_goal_safe_branch_aux_probe_P300_U220_L5e6_D230_Decay250_best_epoch_265.zip",
        "note": "Current balanced candidate: Decay250 best epoch 265.",
    },
    {
        "tag": "U220_D230_epoch290",
        "checkpoint": ROOT_DIR
        / "train_log"
        / "baseline_attn_goal_safe_branch_aux_probe_P300_U220_L5e6_D230_20260425_224122"
        / "checkpoints"
        / "baseline_policy_attn_goal_safe_branch_aux_probe_P300_U220_L5e6_D230_epoch_290.zip",
        "note": "Low-collision candidate from U220_L5e6_D230 epoch 290.",
    },
    {
        "tag": "TTC30_Warn5_epoch260",
        "checkpoint": ROOT_DIR
        / "train_log"
        / "baseline_attn_goal_safe_branch_aux_probe_P300_TTC30_Warn5_20260425_204316"
        / "checkpoints"
        / "baseline_policy_attn_goal_safe_branch_aux_probe_P300_TTC30_Warn5_best_epoch_260.zip",
        "note": "Soft-warning TTC candidate, best epoch 260.",
    },
]


DETAIL_FIELDS = [
    "model_tag",
    "traj_index",
    "filename",
    "steps",
    "terminated",
    "truncated",
    "merge_success",
    "endpoint_success",
    "safety_success",
    "collision",
    "failure_type",
    "episode_reward",
    "eval_dense_return",
    "paper_score",
    "mean_speed_ftps",
    "mean_speed_mps",
    "mean_abs_acc_ftps2",
    "mean_abs_jerk_mps3",
    "min_ttc",
    "min_ttc_step",
    "min_ttc_time_s",
    "min_thw",
    "min_thw_step",
    "min_thw_time_s",
    "first_collision_step",
    "first_collision_time_s",
    "collision_px",
    "collision_py",
    "collision_speed_ftps",
    "collision_action_x",
    "collision_action_y",
    "collision_ax_ftps2",
    "collision_ay_ftps2",
    "collision_nearest_vehicle",
    "collision_nearest_dx",
    "collision_nearest_dy",
    "collision_nearest_rel_vy",
    "min_ttc_px",
    "min_ttc_py",
    "min_ttc_action_x",
    "min_ttc_action_y",
    "min_ttc_nearest_vehicle",
    "min_ttc_nearest_dx",
    "min_ttc_nearest_dy",
    "min_ttc_nearest_rel_vy",
    "final_px",
    "final_py",
    "final_vx",
    "final_vy",
    "dist_to_goal",
    "in_target_lane_final",
    "in_aux_lane_final",
    "expert_len",
]


SUMMARY_FIELDS = [
    "model_tag",
    "checkpoint",
    "total",
    "merge_success_rate",
    "endpoint_success_rate",
    "safety_success_rate",
    "collision_rate",
    "endpoint_fail_rate",
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
    "note",
]


def load_dataset(device):
    data_paths = [
        ROOT_DIR / "data" / "lane_change_trajectories-0750am-0805am",
        ROOT_DIR / "data" / "lane_change_trajectories-0805am-0820am",
        ROOT_DIR / "data" / "lane_change_trajectories-0820am-0835am",
    ]
    return MergingDataset([str(p) for p in data_paths], device=device)


def nearest_vehicle_snapshot(env, px, py, vy):
    surr = env._get_surround_at_t(env.t)
    names = ["L6_lead", "L5_lead", "L5_follow", "L6_follow"]
    best = {
        "vehicle": "",
        "dx": np.nan,
        "dy": np.nan,
        "rel_vy": np.nan,
    }
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


def classify_failure(endpoint_success, safety_success, collided, truncated):
    if endpoint_success and safety_success:
        return "success"
    labels = []
    if collided:
        labels.append("collision")
    if not endpoint_success:
        labels.append("endpoint_fail")
    if truncated and not endpoint_success:
        labels.append("timeout_or_stuck")
    if not labels and not safety_success:
        labels.append("safety_fail")
    return "+".join(labels)


def paper_score(speed_y_mps_trace, jerk_trace_mps3, ttc_trace, endpoint_success, collided):
    if not speed_y_mps_trace or not jerk_trace_mps3 or not ttc_trace:
        score = -10.0
    else:
        score = (
            float(np.mean(speed_y_mps_trace))
            - float(np.std(speed_y_mps_trace))
            + float(np.mean(np.clip(ttc_trace, 0.0, 20.0)))
            - 5.0 * float(np.mean(np.abs(jerk_trace_mps3)))
        )
    if not endpoint_success:
        score -= 5.0
    if collided:
        score -= 5.0
    return float(score)


def evaluate_single_trajectory(model, dataset, cfg, traj_index):
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

    while not (terminated or truncated) and step_count < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        obs, reward, terminated, truncated, info = env.step(action)
        last_info = dict(info)

        px, py, vx, vy = [float(x) for x in env.ego_state]
        speed = float(np.hypot(vx, vy))
        ax_phys = float(action[0] * cfg.PHYS_STEER_MAX)
        ay_phys = float(action[1] * cfg.PHYS_ACC_MAX)
        min_ttc = float(info.get("eval_min_ttc", 20.0))
        min_thw = float(info.get("eval_min_thw", 10.0))
        nearest = nearest_vehicle_snapshot(env, px, py, vy)

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

        collided_now = bool(info.get("is_collided", False))
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

    collided = bool(getattr(env, "has_collided_this_episode", False))
    endpoint_success = bool(last_info.get("is_endpoint_success", False))
    safety_success = bool(last_info.get("is_safety_success", False))
    merge_success = bool(last_info.get("is_merge_success", False))
    p_score = paper_score(speed_y_mps_trace, jerk_trace_mps3, ttc_trace, endpoint_success, collided)

    collision_nearest = first_collision["nearest"] if first_collision else {}
    min_ttc_nearest = min_ttc_info["nearest"] if min_ttc_info else {}

    return {
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
    }


def summarize_rows(model_info, rows):
    total = len(rows)
    if total == 0:
        raise ValueError("No evaluation rows to summarize.")

    def mean(key):
        return float(np.mean([float(r[key]) for r in rows]))

    def rate(key):
        return float(np.mean([1.0 if r[key] else 0.0 for r in rows]))

    endpoint_fail_count = sum(1 for r in rows if not r["endpoint_success"])
    collision_count = sum(1 for r in rows if r["collision"])
    safety_fail_count = sum(1 for r in rows if not r["safety_success"])

    return {
        "model_tag": model_info["tag"],
        "checkpoint": str(model_info["checkpoint"]),
        "total": total,
        "merge_success_rate": rate("merge_success"),
        "endpoint_success_rate": rate("endpoint_success"),
        "safety_success_rate": rate("safety_success"),
        "collision_rate": rate("collision"),
        "endpoint_fail_rate": endpoint_fail_count / total,
        "collision_count": collision_count,
        "endpoint_fail_count": endpoint_fail_count,
        "safety_fail_count": safety_fail_count,
        "mean_episode_reward": mean("episode_reward"),
        "mean_eval_dense_return": mean("eval_dense_return"),
        "mean_paper_score": mean("paper_score"),
        "mean_steps": mean("steps"),
        "mean_speed_mps": mean("mean_speed_mps"),
        "mean_abs_jerk_mps3": mean("mean_abs_jerk_mps3"),
        "mean_min_ttc": mean("min_ttc"),
        "mean_min_thw": mean("min_thw"),
        "ttc_lt_3_rate": float(np.mean([1.0 if float(r["min_ttc"]) < 3.0 else 0.0 for r in rows])),
        "ttc_lt_4_rate": float(np.mean([1.0 if float(r["min_ttc"]) < 4.0 else 0.0 for r in rows])),
        "thw_lt_0p5_rate": float(np.mean([1.0 if float(r["min_thw"]) < 0.5 else 0.0 for r in rows])),
        "note": model_info.get("note", ""),
    }


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_overlap(output_dir, all_rows_by_model):
    filenames = sorted({row["filename"] for rows in all_rows_by_model.values() for row in rows})
    tags = list(all_rows_by_model.keys())
    by_tag_name = {
        tag: {row["filename"]: row for row in rows}
        for tag, rows in all_rows_by_model.items()
    }
    rows = []
    for filename in filenames:
        row = {"filename": filename}
        collision_tags = []
        endpoint_fail_tags = []
        safety_fail_tags = []
        for tag in tags:
            item = by_tag_name[tag].get(filename)
            if item is None:
                row[f"{tag}_collision"] = ""
                row[f"{tag}_endpoint_success"] = ""
                row[f"{tag}_safety_success"] = ""
                row[f"{tag}_min_ttc"] = ""
                continue
            row[f"{tag}_collision"] = item["collision"]
            row[f"{tag}_endpoint_success"] = item["endpoint_success"]
            row[f"{tag}_safety_success"] = item["safety_success"]
            row[f"{tag}_min_ttc"] = item["min_ttc"]
            if item["collision"]:
                collision_tags.append(tag)
            if not item["endpoint_success"]:
                endpoint_fail_tags.append(tag)
            if not item["safety_success"]:
                safety_fail_tags.append(tag)
        row["collision_models"] = ";".join(collision_tags)
        row["endpoint_fail_models"] = ";".join(endpoint_fail_tags)
        row["safety_fail_models"] = ";".join(safety_fail_tags)
        row["collision_model_count"] = len(collision_tags)
        row["endpoint_fail_model_count"] = len(endpoint_fail_tags)
        row["safety_fail_model_count"] = len(safety_fail_tags)
        rows.append(row)

    fieldnames = ["filename"]
    for tag in tags:
        fieldnames.extend([
            f"{tag}_collision",
            f"{tag}_endpoint_success",
            f"{tag}_safety_success",
            f"{tag}_min_ttc",
        ])
    fieldnames.extend([
        "collision_models",
        "endpoint_fail_models",
        "safety_fail_models",
        "collision_model_count",
        "endpoint_fail_model_count",
        "safety_fail_model_count",
    ])
    write_csv(output_dir / "failure_case_overlap.csv", rows, fieldnames)


def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = ROOT_DIR / "train_log" / f"failure_case_full_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Failure-case targeted full evaluation")
    print(f"Output: {output_dir}")
    print(f"Device: {device}")
    print("=" * 80)

    for model_info in TARGET_MODELS:
        if not model_info["checkpoint"].exists():
            raise FileNotFoundError(f"Missing checkpoint: {model_info['checkpoint']}")

    dataset = load_dataset(device)
    total = len(dataset)
    print(f"Loaded dataset: {total} trajectories")

    all_summaries = []
    all_rows_by_model = {}

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_size": total,
        "collision_margin": 1.0,
        "target_models": [
            {
                "tag": item["tag"],
                "checkpoint": str(item["checkpoint"]),
                "note": item.get("note", ""),
            }
            for item in TARGET_MODELS
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for model_info in TARGET_MODELS:
        tag = model_info["tag"]
        print("\n" + "-" * 80)
        print(f"Evaluating {tag}")
        print(model_info["checkpoint"])
        model = PPO.load(str(model_info["checkpoint"]), device=device)

        rows = []
        for traj_index in tqdm(range(total), desc=tag):
            row = evaluate_single_trajectory(model, dataset, cfg, traj_index)
            row["model_tag"] = tag
            rows.append(row)

        summary = summarize_rows(model_info, rows)
        all_summaries.append(summary)
        all_rows_by_model[tag] = rows

        detail_path = output_dir / f"{tag}_trajectory_details.csv"
        failures_path = output_dir / f"{tag}_failure_cases.csv"
        collision_path = output_dir / f"{tag}_collision_cases.csv"
        write_csv(detail_path, rows, DETAIL_FIELDS)
        write_csv(
            failures_path,
            [r for r in rows if r["failure_type"] != "success"],
            DETAIL_FIELDS,
        )
        write_csv(
            collision_path,
            [r for r in rows if r["collision"]],
            DETAIL_FIELDS,
        )

        print(
            f"{tag}: endpoint={summary['endpoint_success_rate']:.3f}, "
            f"safety={summary['safety_success_rate']:.3f}, "
            f"collision={summary['collision_rate']:.3f}, "
            f"merge={summary['merge_success_rate']:.3f}, "
            f"ttc={summary['mean_min_ttc']:.3f}, thw={summary['mean_min_thw']:.3f}"
        )

    write_csv(output_dir / "summary.csv", all_summaries, SUMMARY_FIELDS)
    write_overlap(output_dir, all_rows_by_model)

    print("\n" + "=" * 80)
    print("Summary")
    for summary in all_summaries:
        print(
            f"{summary['model_tag']}: "
            f"N={summary['total']}, endpoint={summary['endpoint_success_rate']:.3f}, "
            f"safety={summary['safety_success_rate']:.3f}, "
            f"collision={summary['collision_rate']:.3f}, "
            f"collisions={summary['collision_count']}, "
            f"endpoint_fail={summary['endpoint_fail_count']}"
        )
    print(f"Saved: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
