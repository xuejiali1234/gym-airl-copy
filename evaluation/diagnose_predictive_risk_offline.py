from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from stable_baselines3 import PPO
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from configs.config import Config  # noqa: E402
from envs.merging_env import MergingEnv  # noqa: E402
from evaluation.failure_case_full_evaluate import TARGET_MODELS, load_dataset  # noqa: E402
from model.attention_net import AttentionFeaturesExtractor, GoalConditionedMLPFeaturesExtractor  # noqa: F401,E402
from model.predictive_safety_oracle import PredictiveSafetyOracle  # noqa: E402
from model.safety_pretrain_q import _generate_predictive_candidate_actions  # noqa: E402


DEFAULT_HARD_CASE_LIST = (
    ROOT_DIR / "train_log" / "failure_case_full_eval_20260427_205244" / "common15_trajectory_summary_for_gpt.csv"
)
DEFAULT_MODEL_TAG = "U220_D230_epoch290"


@dataclass
class GroupResult:
    group: str
    n_samples: int
    risk_mean: float
    risk_p50: float
    risk_p90: float
    risk_p99: float
    critical_rate: float
    risk_gt_07_rate: float


class SingleTrajDataset:
    def __init__(self, traj, expert_mean, expert_std):
        self.trajectories = [traj]
        self.expert_mean = expert_mean
        self.expert_std = expert_std
        self.confidence_weights = np.ones(1, dtype=np.float32)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.trajectories[idx]


def parse_args():
    parser = argparse.ArgumentParser(description="Offline predictive risk diagnosis for the current oracle.")
    parser.add_argument("--model-tag", default=DEFAULT_MODEL_TAG, help="Model tag from TARGET_MODELS.")
    parser.add_argument("--model-path", default="", help="Optional explicit checkpoint path.")
    parser.add_argument("--hard-list", default=str(DEFAULT_HARD_CASE_LIST), help="Common15 filename CSV.")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory. Defaults to train_log/predictive_risk_offline_diag_<timestamp>.",
    )
    parser.add_argument("--max-expert-samples", type=int, default=0, help="Optional cap for expert-state samples.")
    parser.add_argument(
        "--oracle-batch-size",
        type=int,
        default=8192,
        help="Chunk size when evaluating predictive oracle in batches.",
    )
    parser.add_argument(
        "--policy-batch-size",
        type=int,
        default=4096,
        help="Batch size when querying PPO on expert observations.",
    )
    parser.add_argument("--seed", type=int, default=44)
    return parser.parse_args()


def resolve_model_path(args) -> Path:
    if args.model_path:
        path = Path(args.model_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {path}")
        return path

    for item in TARGET_MODELS:
        if item["tag"] == args.model_tag:
            return Path(item["checkpoint"]).resolve()
    raise ValueError(f"Unknown model tag {args.model_tag!r}; available: {[item['tag'] for item in TARGET_MODELS]}")


def load_hard_filenames(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "filename" not in reader.fieldnames:
            raise ValueError(f"Hard-case list must contain a filename column: {path}")
        filenames = [row["filename"] for row in reader if row.get("filename")]
    filenames = sorted(dict.fromkeys(filenames))
    if not filenames:
        raise ValueError(f"No filenames found in hard-case list: {path}")
    return filenames


def build_dataset_index_map(dataset) -> dict[str, int]:
    out = {}
    for idx, traj in enumerate(dataset.trajectories):
        out[traj.get("filename", f"trajectory_{idx}.csv")] = idx
    return out


def maybe_subsample(states: np.ndarray, actions: np.ndarray, obs: np.ndarray, max_samples: int, seed: int):
    if max_samples <= 0 or len(states) <= max_samples:
        return states, actions, obs
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(states), size=max_samples, replace=False)
    idx = np.sort(idx)
    return states[idx], actions[idx], obs[idx]


def build_expert_arrays(dataset, cfg: Config):
    states = []
    actions = []
    observations = []
    enable_goal = getattr(cfg, "ENABLE_GOAL_CONDITION", False)
    for traj in dataset.trajectories:
        traj_states = np.asarray(traj["state"], dtype=np.float32)
        traj_actions = np.asarray(traj["action"], dtype=np.float32)
        if enable_goal:
            traj_obs = np.concatenate([traj_states, np.asarray(traj["goal"], dtype=np.float32)], axis=-1)
        else:
            traj_obs = traj_states.copy()
        states.append(traj_states)
        actions.append(traj_actions)
        observations.append(traj_obs)
    return np.concatenate(states, axis=0), np.concatenate(actions, axis=0), np.concatenate(observations, axis=0)


def predict_actions_batched(model: PPO, observations: np.ndarray, batch_size: int) -> np.ndarray:
    actions = []
    for start in range(0, len(observations), batch_size):
        obs_batch = observations[start : start + batch_size]
        pred, _ = model.predict(obs_batch, deterministic=True)
        pred = np.asarray(pred, dtype=np.float32)
        if pred.ndim == 1:
            pred = pred.reshape(1, -1)
        actions.append(pred)
    return np.concatenate(actions, axis=0)


def analyze_oracle_in_chunks(
    oracle: PredictiveSafetyOracle,
    states: np.ndarray,
    actions: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    risk_scores = []
    critical = []
    with torch.no_grad():
        for start in range(0, len(states), batch_size):
            batch_states = torch.as_tensor(states[start : start + batch_size], dtype=torch.float32)
            batch_actions = torch.as_tensor(actions[start : start + batch_size], dtype=torch.float32)
            info = oracle.analyze_batch(batch_states, batch_actions)
            risk_scores.append(info["risk_score"].cpu().numpy().reshape(-1))
            critical.append(info["critical_label"].cpu().numpy().reshape(-1))
    return np.concatenate(risk_scores, axis=0), np.concatenate(critical, axis=0)


def summarize_group(name: str, risks: np.ndarray, critical: np.ndarray) -> GroupResult:
    if len(risks) == 0:
        return GroupResult(name, 0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
    return GroupResult(
        group=name,
        n_samples=int(len(risks)),
        risk_mean=float(np.mean(risks)),
        risk_p50=float(np.percentile(risks, 50)),
        risk_p90=float(np.percentile(risks, 90)),
        risk_p99=float(np.percentile(risks, 99)),
        critical_rate=float(np.mean(critical >= 0.5)),
        risk_gt_07_rate=float(np.mean(risks > 0.7)),
    )


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_rollout_groups(
    model: PPO,
    dataset,
    cfg: Config,
    hard_indices: set[int],
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    hard_rows = []
    normal_states = []
    normal_actions = []

    for traj_index, traj in enumerate(tqdm(dataset.trajectories, desc="Rollout diagnosis")):
        single_dataset = SingleTrajDataset(traj, dataset.expert_mean, dataset.expert_std)
        env = MergingEnv(single_dataset)
        env.collision_margin = 1.0

        obs, _ = env.reset(seed=cfg.SEED + traj_index)
        terminated = False
        truncated = False
        max_steps = len(env.current_traj["ego_pos"]) + 50

        episode_obs = []
        episode_actions = []
        step_count = 0
        first_collision_step = None
        final_info = {}

        while not (terminated or truncated) and step_count < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            episode_obs.append(np.asarray(obs, dtype=np.float32).copy())
            episode_actions.append(action.copy())
            obs, _, terminated, truncated, info = env.step(action)
            final_info = dict(info)
            if first_collision_step is None and bool(info.get("is_collided", False) or getattr(env, "has_collided_this_episode", False)):
                first_collision_step = step_count
            step_count += 1

        collided = bool(final_info.get("is_collided", False) or getattr(env, "has_collided_this_episode", False))
        endpoint_success = bool(final_info.get("is_endpoint_success", False))

        if traj_index in hard_indices:
            if first_collision_step is not None:
                start_idx = max(0, first_collision_step - 9)
                window_states = np.asarray(episode_obs[start_idx : first_collision_step + 1], dtype=np.float32)
                window_actions = np.asarray(episode_actions[start_idx : first_collision_step + 1], dtype=np.float32)
            else:
                window_states = np.empty((0, env.observation_space.shape[0]), dtype=np.float32)
                window_actions = np.empty((0, 2), dtype=np.float32)
            hard_rows.append(
                {
                    "traj_index": traj_index,
                    "filename": traj.get("filename", f"trajectory_{traj_index}.csv"),
                    "collided": collided,
                    "endpoint_success": endpoint_success,
                    "first_collision_step": "" if first_collision_step is None else int(first_collision_step),
                    "window_sample_count": int(len(window_states)),
                    "_states": window_states,
                    "_actions": window_actions,
                }
            )

        if (not collided) and endpoint_success and episode_obs:
            normal_states.append(np.asarray(episode_obs, dtype=np.float32))
            normal_actions.append(np.asarray(episode_actions, dtype=np.float32))

    if normal_states:
        full_normal_states = np.concatenate(normal_states, axis=0)
        full_normal_actions = np.concatenate(normal_actions, axis=0)
    else:
        full_normal_states = np.empty((0, 18 if getattr(cfg, "ENABLE_GOAL_CONDITION", False) else 16), dtype=np.float32)
        full_normal_actions = np.empty((0, 2), dtype=np.float32)
    return hard_rows, full_normal_states, full_normal_actions


def flatten_hard_rows(hard_rows: Sequence[dict]) -> tuple[np.ndarray, np.ndarray]:
    states = []
    actions = []
    for row in hard_rows:
        if row["_states"].size == 0:
            continue
        states.append(row["_states"])
        actions.append(row["_actions"])
    if not states:
        return np.empty((0, 18), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
    return np.concatenate(states, axis=0), np.concatenate(actions, axis=0)


def strip_private_fields(rows: Iterable[dict]) -> list[dict]:
    cleaned = []
    for row in rows:
        cleaned.append({k: v for k, v in row.items() if not k.startswith("_")})
    return cleaned


def main():
    args = parse_args()
    cfg = Config()
    device = "cpu"

    checkpoint_path = resolve_model_path(args)
    hard_list_path = Path(args.hard_list).expanduser().resolve()
    hard_filenames = load_hard_filenames(hard_list_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else ROOT_DIR / "train_log" / f"predictive_risk_offline_diag_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(device=device)
    model = PPO.load(str(checkpoint_path), device=device)
    oracle = PredictiveSafetyOracle(
        cfg,
        dataset.expert_mean,
        dataset.expert_std,
        horizon_steps=int(getattr(cfg, "PREDICTIVE_SAFETY_HORIZON_STEPS", 10)),
        dt=float(getattr(cfg, "PREDICTIVE_SAFETY_DT", cfg.DT)),
    )

    filename_to_index = build_dataset_index_map(dataset)
    missing = [name for name in hard_filenames if name not in filename_to_index]
    if missing:
        raise ValueError(f"Hard-case filenames missing from dataset: {missing}")
    hard_indices = {filename_to_index[name] for name in hard_filenames}

    expert_states, expert_actions, expert_obs = build_expert_arrays(dataset, cfg)
    expert_states, expert_actions, expert_obs = maybe_subsample(
        expert_states,
        expert_actions,
        expert_obs,
        max_samples=args.max_expert_samples,
        seed=args.seed,
    )

    candidate_states_t, candidate_actions_t = _generate_predictive_candidate_actions(
        torch.as_tensor(expert_states, dtype=torch.float32),
        torch.as_tensor(expert_actions, dtype=torch.float32),
        generator=torch.Generator().manual_seed(args.seed),
    )
    candidate_states = candidate_states_t.cpu().numpy()
    candidate_actions = candidate_actions_t.cpu().numpy()

    print("[*] Predicting policy actions on expert observations...")
    policy_actions = predict_actions_batched(model, expert_obs, batch_size=args.policy_batch_size)

    print("[*] Rolling out policy on full217 / hard15 splits...")
    hard_rows, normal_states, normal_actions = collect_rollout_groups(model, dataset, cfg, hard_indices)
    hard_states, hard_actions = flatten_hard_rows(hard_rows)

    groups = [
        ("expert_action", expert_states, expert_actions),
        ("synthetic_candidate_action", candidate_states, candidate_actions),
        ("policy_action_on_expert_state", expert_states, policy_actions),
        ("hard15_pre_collision_1s", hard_states, hard_actions),
        ("full217_noncollision_normal", normal_states, normal_actions),
    ]

    summary_rows = []
    detailed_hard_rows = []

    for name, states, actions in groups:
        print(f"[*] Oracle analysis: {name} ({len(states)} samples)")
        risks, critical = analyze_oracle_in_chunks(
            oracle,
            states,
            actions,
            batch_size=args.oracle_batch_size,
        )
        summary = summarize_group(name, risks, critical)
        summary_rows.append(summary.__dict__)

        if name == "hard15_pre_collision_1s":
            cursor = 0
            for row in hard_rows:
                n = row["window_sample_count"]
                if n > 0:
                    local_risk = risks[cursor : cursor + n]
                    local_critical = critical[cursor : cursor + n]
                    cursor += n
                    row.update(
                        {
                            "risk_mean": float(np.mean(local_risk)),
                            "risk_p50": float(np.percentile(local_risk, 50)),
                            "risk_p90": float(np.percentile(local_risk, 90)),
                            "risk_p99": float(np.percentile(local_risk, 99)),
                            "critical_rate": float(np.mean(local_critical >= 0.5)),
                            "risk_gt_07_rate": float(np.mean(local_risk > 0.7)),
                        }
                    )
                else:
                    row.update(
                        {
                            "risk_mean": float("nan"),
                            "risk_p50": float("nan"),
                            "risk_p90": float("nan"),
                            "risk_p99": float("nan"),
                            "critical_rate": float("nan"),
                            "risk_gt_07_rate": float("nan"),
                        }
                    )
            detailed_hard_rows = strip_private_fields(hard_rows)

    fieldnames = [
        "group",
        "n_samples",
        "risk_mean",
        "risk_p50",
        "risk_p90",
        "risk_p99",
        "critical_rate",
        "risk_gt_07_rate",
    ]
    write_csv(output_dir / "predictive_risk_group_summary.csv", summary_rows, fieldnames)

    if detailed_hard_rows:
        hard_fields = [
            "traj_index",
            "filename",
            "collided",
            "endpoint_success",
            "first_collision_step",
            "window_sample_count",
            "risk_mean",
            "risk_p50",
            "risk_p90",
            "risk_p99",
            "critical_rate",
            "risk_gt_07_rate",
        ]
        write_csv(output_dir / "hard15_precollision_file_summary.csv", detailed_hard_rows, hard_fields)

    summary_json = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": str(checkpoint_path),
        "hard_list": str(hard_list_path),
        "oracle_config": {
            "horizon_steps": int(oracle.horizon_steps),
            "dt": float(oracle.dt),
            "ttc_lead_threshold": float(oracle.ttc_lead_threshold),
            "ttc_follow_threshold": float(oracle.ttc_follow_threshold),
            "thw_lead_threshold": float(oracle.thw_lead_threshold),
            "thw_follow_threshold": float(oracle.thw_follow_threshold),
            "gap_lead_threshold": float(oracle.gap_lead_threshold),
            "gap_follow_threshold": float(oracle.gap_follow_threshold),
            "gap_lead_tau": float(oracle.gap_lead_tau),
            "gap_follow_tau": float(oracle.gap_follow_tau),
            "target_lane_background_weight": float(oracle.target_lane_background_weight),
            "target_ttc_thw_weight": float(oracle.target_ttc_thw_weight),
            "target_gap_weight": float(oracle.target_gap_weight),
            "current_lane_risk_weight": float(oracle.current_lane_risk_weight),
        },
        "group_summary": summary_rows,
        "interpretation": {
            "hard15_precollision_recall_risk_gt_07": next(
                (row["risk_gt_07_rate"] for row in summary_rows if row["group"] == "hard15_pre_collision_1s"),
                None,
            ),
            "hard15_precollision_critical_rate": next(
                (row["critical_rate"] for row in summary_rows if row["group"] == "hard15_pre_collision_1s"),
                None,
            ),
            "normal_safe_false_positive_risk_gt_07": next(
                (row["risk_gt_07_rate"] for row in summary_rows if row["group"] == "full217_noncollision_normal"),
                None,
            ),
            "normal_safe_critical_false_positive": next(
                (row["critical_rate"] for row in summary_rows if row["group"] == "full217_noncollision_normal"),
                None,
            ),
        },
    }
    with (output_dir / "predictive_risk_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)

    print(f"[*] Saved offline predictive risk diagnosis to: {output_dir}")
    for row in summary_rows:
        print(
            f"    - {row['group']}: n={row['n_samples']}, "
            f"risk_mean={row['risk_mean']:.4f}, p90={row['risk_p90']:.4f}, "
            f"critical_rate={row['critical_rate']:.4f}, risk>0.7={row['risk_gt_07_rate']:.4f}"
        )


if __name__ == "__main__":
    main()
