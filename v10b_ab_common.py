import copy
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from stable_baselines3 import PPO

from configs.config import Config
from evaluation.failure_case_full_evaluate import evaluate_single_trajectory, summarize_rows
from evaluation.hard_case_protocol_evaluate import annotate_summary
from evaluation.safety_shield_evaluate import evaluate_single_trajectory_with_shield, shield_action, summarize_with_shield
from model.attention_net import AttentionFeaturesExtractor, GoalConditionedMLPFeaturesExtractor  # noqa: F401
from v10b_distill_common import BASE_MODEL_TAG, get_model_info_by_tag, load_teacher_payload


ROOT_DIR = Path(__file__).resolve().parent
INPUT_DIM = 18 + 2 + 4
RISK_SIDE_LABELS = ("lead", "follow", "none")
RISK_REASON_LABELS = ("low_ttc", "low_thw", "overlap", "recovery", "veto", "none")


def split_dataset_like_train(dataset, train_ratio=0.8, seed=42):
    num_trajs = len(dataset.trajectories)
    if num_trajs < 2:
        return dataset, dataset

    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_trajs)
    split_idx = max(1, min(num_trajs - 1, int(num_trajs * train_ratio)))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    def make_subset(idxs):
        subset = copy.copy(dataset)
        subset.trajectories = [dataset.trajectories[int(i)] for i in idxs]
        if hasattr(dataset, "confidence_weights"):
            subset.confidence_weights = np.asarray(dataset.confidence_weights)[idxs].copy()
        return subset

    return make_subset(train_indices), make_subset(val_indices)


def sanitize_numeric(array: np.ndarray, posinf=1e3, neginf=-1e3) -> np.ndarray:
    return np.nan_to_num(array.astype(np.float32), nan=0.0, posinf=posinf, neginf=neginf)


def build_ab_input_matrix(samples: Sequence[Dict]) -> np.ndarray:
    obs = np.stack([sanitize_numeric(np.asarray(item["obs"], dtype=np.float32)) for item in samples], axis=0)
    policy_action = np.stack(
        [sanitize_numeric(np.asarray(item["policy_action"], dtype=np.float32), posinf=1.0, neginf=-1.0) for item in samples],
        axis=0,
    )
    aux = np.stack(
        [
            sanitize_numeric(
                np.asarray(
                    [
                        item.get("lead_gap", 0.0),
                        item.get("lead_thw", 0.0),
                        item.get("follow_gap", 0.0),
                        item.get("follow_thw", 0.0),
                    ],
                    dtype=np.float32,
                )
            )
            for item in samples
        ],
        axis=0,
    )
    return np.concatenate([obs, policy_action, aux], axis=1).astype(np.float32)


def teacher_split_indices(samples: Sequence[Dict], seed: int) -> Dict[str, np.ndarray]:
    traj_to_indices: Dict[int, List[int]] = {}
    traj_is_hard: Dict[int, bool] = {}
    for idx, sample in enumerate(samples):
        traj_index = int(sample["traj_index"])
        traj_to_indices.setdefault(traj_index, []).append(idx)
        traj_is_hard[traj_index] = bool(sample.get("split") == "hard15")

    hard_trajs = sorted([traj for traj, flag in traj_is_hard.items() if flag])
    regular_trajs = sorted([traj for traj, flag in traj_is_hard.items() if not flag])

    rng = np.random.default_rng(seed)
    regular_perm = rng.permutation(regular_trajs) if regular_trajs else np.asarray([], dtype=np.int64)
    split_idx = int(len(regular_perm) * 0.8)
    split_idx = max(1, split_idx) if len(regular_perm) > 1 else len(regular_perm)
    split_idx = min(split_idx, max(len(regular_perm) - 1, 0)) if len(regular_perm) > 1 else split_idx
    train_trajs = set(hard_trajs)
    train_trajs.update(int(x) for x in regular_perm[:split_idx])
    val_trajs = set(int(x) for x in regular_perm[split_idx:])

    train_indices = [idx for traj in train_trajs for idx in traj_to_indices.get(traj, [])]
    val_indices = [idx for traj in val_trajs for idx in traj_to_indices.get(traj, [])]
    hard_indices = [idx for traj in hard_trajs for idx in traj_to_indices.get(traj, [])]
    full_indices = list(range(len(samples)))

    return {
        "train": np.asarray(sorted(train_indices), dtype=np.int64),
        "val": np.asarray(sorted(val_indices), dtype=np.int64),
        "hard": np.asarray(sorted(hard_indices), dtype=np.int64),
        "full": np.asarray(full_indices, dtype=np.int64),
        "hard_trajs": np.asarray(hard_trajs, dtype=np.int64),
    }


def make_classification_arrays(samples: Sequence[Dict]) -> Dict[str, np.ndarray]:
    return {
        "intervene": np.asarray([float(bool(item.get("shield_intervened", False))) for item in samples], dtype=np.float32),
        "critical": np.asarray([float(bool(item.get("critical_risk", False))) for item in samples], dtype=np.float32),
        "risk_side": np.asarray([RISK_SIDE_LABELS.index(item.get("risk_side_label", "none")) for item in samples], dtype=np.int64),
        "risk_reason": np.asarray([RISK_REASON_LABELS.index(item.get("risk_reason_label", "none")) for item in samples], dtype=np.int64),
        "warning": np.asarray([bool(item.get("shield_warning", False)) for item in samples], dtype=np.bool_),
        "hard15": np.asarray([bool(item.get("split") == "hard15") for item in samples], dtype=np.bool_),
        "normal_safe": np.asarray([bool(item.get("normal_safe_mask", False)) for item in samples], dtype=np.bool_),
        "delta_action": np.stack(
            [sanitize_numeric(np.asarray(item.get("delta_action", [0.0, 0.0]), dtype=np.float32), posinf=1.0, neginf=-1.0) for item in samples],
            axis=0,
        ).astype(np.float32),
    }


def binary_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    positives = y_true > 0.5
    denom = max(int(positives.sum()), 1)
    tp = int(((y_pred > 0.5) & positives).sum())
    return float(tp / denom)


def binary_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    predicted = y_pred > 0.5
    denom = max(int(predicted.sum()), 1)
    tp = int((predicted & (y_true > 0.5)).sum())
    return float(tp / denom)


def multiclass_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    return float((y_true == y_pred).sum() / len(y_true))


def append_csv_row(path: Path, row: Dict, fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fieldnames or row.keys())
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def evaluate_indices_no_shield(model, dataset, cfg, traj_indices: Sequence[int], model_tag=BASE_MODEL_TAG) -> Tuple[List[Dict], Dict]:
    model_info = get_model_info_by_tag(model_tag)
    rows = [{**evaluate_single_trajectory(model, dataset, cfg, traj_index), "model_tag": model_tag} for traj_index in traj_indices]
    summary = summarize_rows(model_info, rows)
    return rows, summary


def evaluate_indices_with_shield(model, dataset, cfg, traj_indices: Sequence[int], shield_params: Dict, split_name: str, thresholds: Dict, model_tag=BASE_MODEL_TAG) -> Tuple[List[Dict], Dict]:
    model_info = get_model_info_by_tag(model_tag)
    rows = [
        {**evaluate_single_trajectory_with_shield(model, dataset, cfg, traj_index, shield_params), "model_tag": model_tag}
        for traj_index in traj_indices
    ]
    summary = summarize_with_shield(model_info, rows, split_name, thresholds)
    return rows, summary


class ClassifierGatePolicyWrapper:
    def __init__(self, base_model: PPO, classifier: torch.nn.Module, device: torch.device, shield_params: Dict, threshold: float = 0.5):
        self.base_model = base_model
        self.classifier = classifier
        self.device = device
        self.shield_params = shield_params
        self.threshold = float(threshold)

    def predict(self, obs, deterministic=True):
        policy_action, _ = self.base_model.predict(obs, deterministic=deterministic)
        policy_action = np.asarray(policy_action, dtype=np.float32).reshape(-1)
        obs_vec = sanitize_numeric(np.asarray(obs, dtype=np.float32))
        aux = np.zeros(4, dtype=np.float32)
        x = np.concatenate([obs_vec, policy_action, aux], axis=0)[None, :]
        x_t = torch.as_tensor(x, device=self.device)
        self.classifier.eval()
        with torch.no_grad():
            logits = self.classifier(x_t)
            intervene_prob = torch.sigmoid(logits["intervene_logit"]).item()
        if intervene_prob >= self.threshold and hasattr(self, "_active_env"):
            shielded_action, _ = shield_action(
                env=self._active_env,
                policy_action=policy_action,
                params=self.shield_params,
                step_count=getattr(self, "_episode_step", 0),
                shield_interventions=getattr(self, "_shield_interventions", 0),
                consecutive_interventions=getattr(self, "_consecutive_interventions", 0),
                safe_steps=getattr(self, "_safe_steps", 0),
            )
            return np.asarray(shielded_action, dtype=np.float32), None
        return policy_action, None

    def bind_env(self, env):
        self._active_env = env
        self._episode_step = 0
        self._shield_interventions = 0
        self._consecutive_interventions = 0
        self._safe_steps = 0


class ResidualPolicyWrapper:
    def __init__(self, base_model: PPO, residual_model: torch.nn.Module, device: torch.device, gate_threshold: float = 0.5):
        self.base_model = base_model
        self.residual_model = residual_model
        self.device = device
        self.gate_threshold = float(gate_threshold)

    def predict(self, obs, deterministic=True):
        policy_action, _ = self.base_model.predict(obs, deterministic=deterministic)
        policy_action = np.asarray(policy_action, dtype=np.float32).reshape(-1)
        obs_vec = sanitize_numeric(np.asarray(obs, dtype=np.float32))
        aux = np.zeros(4, dtype=np.float32)
        x = np.concatenate([obs_vec, policy_action, aux], axis=0)[None, :]
        x_t = torch.as_tensor(x, device=self.device)
        self.residual_model.eval()
        with torch.no_grad():
            out = self.residual_model(x_t)
            gate = torch.sigmoid(out["gate_logit"]).item()
            delta = out["delta_action"].squeeze(0).cpu().numpy()
        if gate < self.gate_threshold:
            gate = 0.0
        action = np.clip(policy_action + gate * delta, -1.0, 1.0).astype(np.float32)
        return action, None


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(__import__("json").dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

