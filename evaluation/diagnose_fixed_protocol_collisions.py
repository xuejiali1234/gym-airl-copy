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
from envs.merging_env import MergingEnv
from model.attention_net import AttentionFeaturesExtractor, GoalConditionedMLPFeaturesExtractor  # noqa: F401
from model.predictive_safety_oracle import PredictiveSafetyOracle
from utils.data_loader import MergingDataset


DEFAULT_HARD_CASE_DIR = ROOT_DIR / "train_log" / "failure_case_full_eval_20260427_205244"
DEFAULT_HARD_CASE_LIST = DEFAULT_HARD_CASE_DIR / "common15_trajectory_summary_for_gpt.csv"


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


CANDIDATE_NAMES = ["policy", "hold", "delay", "decel", "accel", "merge"]
PROGRESS_COMPATIBLE_TYPES = {"policy", "delay", "accel", "merge"}


DIAG_FIELDS = [
    "model_tag",
    "split",
    "checkpoint",
    "traj_index",
    "filename",
    "collision",
    "collision_step",
    "collision_time_s",
    "endpoint_success",
    "merge_success",
    "safety_success",
    "collision_object",
    "collision_nearest_vehicle",
    "collision_nearest_dx",
    "collision_nearest_dy",
    "collision_nearest_rel_vy",
    "pre1s_risk_mean",
    "pre1s_risk_max",
    "policy_action_risk",
    "policy_action_x",
    "policy_action_y",
    "min_risk_action_type",
    "min_risk_action_risk",
    "min_risk_action_x",
    "min_risk_action_y",
    "progress_safe_exists",
    "progress_safe_action_type",
    "progress_safe_action_risk",
    "progress_safe_action_x",
    "progress_safe_action_y",
    "risk_gap",
    "reward_gap",
    "reward_gap_note",
]


SUMMARY_FIELDS = [
    "model_tag",
    "split",
    "checkpoint",
    "total",
    "collision_count",
    "collision_rate",
    "merge_success_rate",
    "endpoint_success_rate",
    "safety_success_rate",
    "mean_pre1s_risk_mean",
    "mean_pre1s_risk_max",
    "mean_policy_action_risk",
    "mean_min_risk_action_risk",
    "mean_risk_gap",
    "progress_safe_exists_rate",
    "lead_collision_count",
    "follow_collision_count",
    "side_collision_count",
    "unknown_collision_count",
    "reward_gap_available",
    "note",
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
    parser = argparse.ArgumentParser(description="No-training fixed-protocol collision diagnosis.")
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="TAG=CHECKPOINT",
        help="Model to diagnose. Can be repeated. Defaults to the four built-in checkpoints.",
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
        help="Output directory. Defaults to train_log/D31_FixedProtocol_CollisionDiagnosis_<timestamp>.",
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


def build_candidate_actions(action_tensor):
    lateral = action_tensor[:, 0]
    longitudinal = action_tensor[:, 1]
    hold_lane = torch.stack([torch.zeros_like(lateral), longitudinal], dim=-1)
    delay_merge = torch.stack([0.5 * lateral.clamp(max=0.0), longitudinal], dim=-1)
    mild_decel = torch.stack([lateral, (longitudinal - 0.25).clamp(-1.0, 1.0)], dim=-1)
    mild_accel = torch.stack([lateral, (longitudinal + 0.25).clamp(-1.0, 1.0)], dim=-1)
    mild_merge = torch.stack([(lateral - 0.25).clamp(-1.0, 1.0), longitudinal], dim=-1)
    candidates = torch.stack(
        [action_tensor, hold_lane, delay_merge, mild_decel, mild_accel, mild_merge],
        dim=1,
    )
    return candidates.clamp(-1.0, 1.0)


def analyze_candidate_set(oracle, obs_before, action):
    state_tensor = torch.as_tensor(obs_before[None, :], dtype=torch.float32)
    action_tensor = torch.as_tensor(action[None, :], dtype=torch.float32)
    candidates = build_candidate_actions(action_tensor)
    state_rep = state_tensor.repeat(candidates.shape[1], 1)
    cand_flat = candidates.reshape(-1, action_tensor.shape[-1])
    analysis = oracle.analyze_batch(state_rep, cand_flat)
    risks = analysis["risk_score"].reshape(-1).cpu().numpy()
    cand_np = candidates[0].cpu().numpy()

    min_idx = int(np.argmin(risks))
    policy_risk = float(risks[0])
    min_risk = float(risks[min_idx])
    min_type = CANDIDATE_NAMES[min_idx]
    min_action = cand_np[min_idx]

    progress_indices = [idx for idx, name in enumerate(CANDIDATE_NAMES) if name in PROGRESS_COMPATIBLE_TYPES and idx != 0]
    progress_better = [idx for idx in progress_indices if float(risks[idx]) + 1e-6 < policy_risk]

    progress_exists = bool(progress_better)
    if progress_better:
        best_progress_idx = min(progress_better, key=lambda idx: float(risks[idx]))
        progress_type = CANDIDATE_NAMES[best_progress_idx]
        progress_risk = float(risks[best_progress_idx])
        progress_action = cand_np[best_progress_idx]
    else:
        best_progress_idx = None
        progress_type = ""
        progress_risk = np.nan
        progress_action = np.array([np.nan, np.nan], dtype=np.float32)

    return {
        "policy_action_risk": policy_risk,
        "min_risk_action_type": min_type,
        "min_risk_action_risk": min_risk,
        "min_risk_action_x": float(min_action[0]),
        "min_risk_action_y": float(min_action[1]),
        "progress_safe_exists": progress_exists,
        "progress_safe_action_type": progress_type,
        "progress_safe_action_risk": progress_risk,
        "progress_safe_action_x": float(progress_action[0]) if not np.isnan(progress_action[0]) else "",
        "progress_safe_action_y": float(progress_action[1]) if not np.isnan(progress_action[1]) else "",
        "risk_gap": float(policy_risk - min_risk),
    }


def evaluate_single_trajectory(model, dataset, cfg, oracle, traj_index):
    traj = dataset[traj_index]
    single_dataset = SingleTrajDataset(traj, dataset.expert_mean, dataset.expert_std)
    env = MergingEnv(single_dataset)
    env.collision_margin = 1.0

    obs, _ = env.reset(seed=cfg.SEED + traj_index)
    terminated = False
    truncated = False
    max_steps = len(env.current_traj["ego_pos"]) + 50

    step_records = []
    first_collision = None
    step_count = 0
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
        policy_risk = float(analysis["risk_score"][0, 0].item())

        obs, reward, terminated, truncated, info = env.step(action)
        last_info = dict(info)

        post_px, post_py, post_vx, post_vy = [float(x) for x in env.ego_state]
        collided_now = bool(info.get("is_collided", False))
        record = {
            "step": step_count,
            "obs_before": obs_before,
            "action": action.copy(),
            "policy_risk": policy_risk,
            "pre_px": pre_px,
            "pre_py": pre_py,
            "pre_vx": pre_vx,
            "pre_vy": pre_vy,
            "post_px": post_px,
            "post_py": post_py,
            "post_vx": post_vx,
            "post_vy": post_vy,
            "nearest_pre": nearest_pre,
            "collided_now": collided_now,
            "reward": float(reward),
        }
        step_records.append(record)

        if first_collision is None and collided_now:
            first_collision = record

        step_count += 1

    collided = bool(getattr(env, "has_collided_this_episode", False))
    endpoint_success = bool(last_info.get("is_endpoint_success", False))
    merge_success = bool(last_info.get("is_merge_success", False))
    safety_success = bool(last_info.get("is_safety_success", False))

    if not collided or first_collision is None:
        return {
            "collision": False,
            "traj_index": traj_index,
            "filename": traj.get("filename", f"trajectory_{traj_index}.csv"),
            "endpoint_success": endpoint_success,
            "merge_success": merge_success,
            "safety_success": safety_success,
        }

    collision_idx = int(first_collision["step"])
    window = step_records[max(0, collision_idx - 9) : collision_idx + 1]
    pre1s_risks = [float(item["policy_risk"]) for item in window]

    candidate_diag = analyze_candidate_set(
        oracle=oracle,
        obs_before=first_collision["obs_before"],
        action=first_collision["action"],
    )
    nearest = first_collision["nearest_pre"]
    collision_object = classify_collision_object(
        nearest_vehicle=nearest.get("vehicle", ""),
        dx=nearest.get("dx", np.nan),
        dy=nearest.get("dy", np.nan),
    )

    return {
        "collision": True,
        "traj_index": traj_index,
        "filename": traj.get("filename", f"trajectory_{traj_index}.csv"),
        "collision_step": collision_idx,
        "collision_time_s": float(collision_idx * cfg.DT),
        "endpoint_success": endpoint_success,
        "merge_success": merge_success,
        "safety_success": safety_success,
        "collision_object": collision_object,
        "collision_nearest_vehicle": nearest.get("vehicle", ""),
        "collision_nearest_dx": nearest.get("dx", ""),
        "collision_nearest_dy": nearest.get("dy", ""),
        "collision_nearest_rel_vy": nearest.get("rel_vy", ""),
        "pre1s_risk_mean": float(np.mean(pre1s_risks)) if pre1s_risks else np.nan,
        "pre1s_risk_max": float(np.max(pre1s_risks)) if pre1s_risks else np.nan,
        "policy_action_risk": float(first_collision["policy_risk"]),
        "policy_action_x": float(first_collision["action"][0]),
        "policy_action_y": float(first_collision["action"][1]),
        "reward_gap": "NA",
        "reward_gap_note": "AIRL discriminator/reward net is not recoverable from PPO-only checkpoint.",
        **candidate_diag,
    }


def summarize_split(model_info, split_name, total, collision_rows):
    object_counts = {"lead": 0, "follow": 0, "side": 0, "unknown": 0}
    for row in collision_rows:
        object_counts[row["collision_object"]] = object_counts.get(row["collision_object"], 0) + 1

    def safe_mean(key):
        vals = [float(row[key]) for row in collision_rows if row.get(key) not in ("", "NA", None)]
        return float(np.mean(vals)) if vals else np.nan

    return {
        "model_tag": model_info["tag"],
        "split": split_name,
        "checkpoint": str(model_info["checkpoint"]),
        "total": total,
        "collision_count": len(collision_rows),
        "collision_rate": len(collision_rows) / total if total else np.nan,
        "merge_success_rate": float(np.mean([1.0 if row["merge_success"] else 0.0 for row in collision_rows])) if collision_rows else np.nan,
        "endpoint_success_rate": float(np.mean([1.0 if row["endpoint_success"] else 0.0 for row in collision_rows])) if collision_rows else np.nan,
        "safety_success_rate": float(np.mean([1.0 if row["safety_success"] else 0.0 for row in collision_rows])) if collision_rows else np.nan,
        "mean_pre1s_risk_mean": safe_mean("pre1s_risk_mean"),
        "mean_pre1s_risk_max": safe_mean("pre1s_risk_max"),
        "mean_policy_action_risk": safe_mean("policy_action_risk"),
        "mean_min_risk_action_risk": safe_mean("min_risk_action_risk"),
        "mean_risk_gap": safe_mean("risk_gap"),
        "progress_safe_exists_rate": float(np.mean([1.0 if row["progress_safe_exists"] else 0.0 for row in collision_rows])) if collision_rows else np.nan,
        "lead_collision_count": object_counts.get("lead", 0),
        "follow_collision_count": object_counts.get("follow", 0),
        "side_collision_count": object_counts.get("side", 0),
        "unknown_collision_count": object_counts.get("unknown", 0),
        "reward_gap_available": False,
        "note": model_info.get("note", ""),
    }


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_report(path, all_rows, all_summaries):
    lines = [
        "# D31 Fixed Protocol Collision Diagnosis",
        "",
        f"Created at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Notes",
        "",
        "- This is a no-training diagnosis.",
        "- `reward_gap` is marked `NA` because PPO checkpoints do not contain AIRL discriminator/reward-net weights.",
        "- The predictive-risk and candidate-action fields are computed offline with the current predictive oracle and current candidate set.",
        "",
        "## Split summary",
        "",
    ]
    for row in all_summaries:
        lines.extend(
            [
                f"### {row['model_tag']} / {row['split']}",
                f"- collisions: {row['collision_count']}/{row['total']} ({row['collision_rate']:.3f})",
                f"- mean pre1s risk mean/max: {row['mean_pre1s_risk_mean']:.3f} / {row['mean_pre1s_risk_max']:.3f}",
                f"- mean policy risk: {row['mean_policy_action_risk']:.3f}",
                f"- mean min-risk action risk: {row['mean_min_risk_action_risk']:.3f}",
                f"- mean risk gap: {row['mean_risk_gap']:.3f}",
                f"- progress-compatible safe action exists rate: {row['progress_safe_exists_rate']:.3f}",
                f"- collision object counts: lead={row['lead_collision_count']}, follow={row['follow_collision_count']}, side={row['side_collision_count']}, unknown={row['unknown_collision_count']}",
                "",
            ]
        )

    lines.extend(["## Heuristic reading", ""])
    if any(row["progress_safe_exists_rate"] > 0.5 for row in all_summaries if not np.isnan(row["progress_safe_exists_rate"])):
        lines.append("- Many collision cases already have a lower-risk progress-compatible candidate, which points more toward regulator/ranking weakness than candidate-set absence.")
    else:
        lines.append("- Progress-compatible safer candidates are rare in the current candidate set, which points more toward candidate-set weakness or safe-action selection logic.")
    if any(row["mean_policy_action_risk"] < 0.4 for row in all_summaries if not np.isnan(row["mean_policy_action_risk"])):
        lines.append("- Some collision cases still receive modest policy risk, which suggests predictive-oracle under-detection is part of the problem.")
    else:
        lines.append("- Collision-causing policy actions are usually scored as clearly risky, so the bigger gap is likely not raw risk visibility.")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8-sig")


def main():
    args = parse_args()
    models = get_models(args)
    hard_list_path, hard_filenames = load_hard_case_filenames(args.hard_list)

    if args.dry_run:
        print("Selected models:")
        for item in models:
            print(f"- {item['tag']}: {item['checkpoint']}")
        print("")
        print(f"Hard-case list: {hard_list_path}")
        print(f"Hard-case filenames ({len(hard_filenames)}):")
        for name in hard_filenames:
            print(f"- {name}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()
    dataset = load_dataset(device=device)
    full_indices = list(range(len(dataset)))
    hard_indices = dataset_indices_by_filename(dataset, hard_filenames)

    oracle = PredictiveSafetyOracle(
        cfg,
        dataset.expert_mean,
        dataset.expert_std,
        horizon_steps=getattr(cfg, "PREDICTIVE_SAFETY_HORIZON_STEPS", 10),
        dt=getattr(cfg, "PREDICTIVE_SAFETY_DT", 0.1),
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (ROOT_DIR / "train_log" / f"D31_FixedProtocol_CollisionDiagnosis_{timestamp}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    all_collision_rows = []
    all_summary_rows = []

    for model_info in models:
        print(f"\n=== Diagnosing {model_info['tag']} ===")
        model = PPO.load(model_info["checkpoint"], device=device)

        for split_name, indices in (("full", full_indices), ("hard15", hard_indices)):
            collision_rows = []
            for traj_index in tqdm(indices, desc=f"{model_info['tag']}:{split_name}"):
                row = evaluate_single_trajectory(model, dataset, cfg, oracle, traj_index)
                if not row["collision"]:
                    continue
                row["model_tag"] = model_info["tag"]
                row["split"] = split_name
                row["checkpoint"] = str(model_info["checkpoint"])
                collision_rows.append({key: row.get(key, "") for key in DIAG_FIELDS})

            all_collision_rows.extend(collision_rows)
            all_summary_rows.append(
                summarize_split(
                    model_info=model_info,
                    split_name=split_name,
                    total=len(indices),
                    collision_rows=collision_rows,
                )
            )

    write_csv(output_dir / "collision_diagnosis.csv", all_collision_rows, DIAG_FIELDS)
    write_csv(output_dir / "collision_diagnosis_summary.csv", all_summary_rows, SUMMARY_FIELDS)
    write_markdown_report(output_dir / "collision_diagnosis_report.md", all_collision_rows, all_summary_rows)

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "hard_case_list": str(hard_list_path),
        "models": [
            {"tag": item["tag"], "checkpoint": str(item["checkpoint"]), "note": item.get("note", "")}
            for item in models
        ],
        "reward_gap_note": "Unavailable from PPO-only checkpoints because AIRL discriminator/reward-net weights are not stored in the policy zip.",
        "output_dir": str(output_dir),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8-sig")

    print("\nDone.")
    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()
