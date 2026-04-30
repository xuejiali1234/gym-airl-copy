import argparse
import copy
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from stable_baselines3 import PPO
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

from configs.config import Config
from envs.merging_env import MergingEnv
from evaluation.failure_case_full_evaluate import (
    evaluate_single_trajectory,
    summarize_rows,
)
from evaluation.hard_case_protocol_evaluate import annotate_summary
from evaluation.safety_shield_evaluate import (
    evaluate_single_trajectory_with_shield,
    shield_action,
    summarize_with_shield,
)
from model.attention_net import AttentionFeaturesExtractor, GoalConditionedMLPFeaturesExtractor  # noqa: F401
from utils.data_loader import MergingDataset
from v10b_distill_common import (
    BASE_MODEL_TAG,
    append_csv_row,
    build_best_key,
    build_v10b_shield_params,
    flatten_eval_summaries,
    get_model_info_by_tag,
    limit_indices,
    load_eval_dataset,
    load_hard_case_info,
    load_teacher_payload,
    resolve_base_checkpoint_from_teacher,
    save_json,
    seed_everything,
)


ROOT_DIR = Path(__file__).resolve().parent
DATA_PATHS = [
    "data/lane_change_trajectories-0750am-0805am",
    "data/lane_change_trajectories-0805am-0820am",
    "data/lane_change_trajectories-0820am-0835am",
]
ALL_METRIC_FIELDS = [
    "hard15_no_shield_merge_success_rate",
    "hard15_no_shield_endpoint_success_rate",
    "hard15_no_shield_safety_success_rate",
    "hard15_no_shield_collision_rate",
    "hard15_no_shield_collision_count",
    "hard15_shield_merge_success_rate",
    "hard15_shield_endpoint_success_rate",
    "hard15_shield_safety_success_rate",
    "hard15_shield_collision_rate",
    "hard15_shield_collision_count",
    "hard15_shield_shield_intervention_rate",
    "full_no_shield_merge_success_rate",
    "full_no_shield_endpoint_success_rate",
    "full_no_shield_safety_success_rate",
    "full_no_shield_collision_rate",
    "full_no_shield_collision_count",
    "full_shield_merge_success_rate",
    "full_shield_endpoint_success_rate",
    "full_shield_safety_success_rate",
    "full_shield_collision_rate",
    "full_shield_collision_count",
    "full_shield_shield_intervention_rate",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run v10b distillation experiments without touching the AIRL main training path."
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=("exp1_actor_only", "exp2_risk_weighted", "exp3_ppo_finetune"),
    )
    parser.add_argument("--teacher-dataset", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--eval-full-every", type=int, default=5)
    parser.add_argument("--eval-full-limit", type=int, default=0)
    parser.add_argument("--eval-hard-limit", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.0)
    parser.add_argument("--rollout-steps", type=int, default=0)
    parser.add_argument("--ppo-update-epochs", type=int, default=0)
    parser.add_argument("--lambda-shield", type=float, default=0.01)
    return parser.parse_args()


def default_output_dir(mode: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT_DIR / "train_log" / f"v10b_distill_{mode}_{timestamp}"


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


def load_train_dataset(device: torch.device, seed: int):
    dataset = MergingDataset(DATA_PATHS, device=device)
    train_dataset, _ = split_dataset_like_train(dataset, train_ratio=0.8, seed=seed)
    return train_dataset


def build_teacher_arrays(samples: List[Dict]) -> Dict[str, np.ndarray]:
    obs = np.stack([np.asarray(item["obs"], dtype=np.float32) for item in samples], axis=0)
    policy_actions = np.stack([np.asarray(item["policy_action"], dtype=np.float32) for item in samples], axis=0)
    shield_actions = np.stack([np.asarray(item["shield_action"], dtype=np.float32) for item in samples], axis=0)
    intervened = np.asarray([bool(item["shield_intervened"]) for item in samples], dtype=np.bool_)
    warning = np.asarray([bool(item["shield_warning"]) for item in samples], dtype=np.bool_)
    hard15 = np.asarray([item["split"] == "hard15" for item in samples], dtype=np.bool_)
    return {
        "obs": obs,
        "policy_actions": policy_actions,
        "shield_actions": shield_actions,
        "intervened": intervened,
        "warning": warning,
        "hard15": hard15,
    }


def state_weight_array(intervened: np.ndarray, warning: np.ndarray) -> np.ndarray:
    normal_safe = ~(intervened | warning)
    out = np.full(intervened.shape, 0.2, dtype=np.float32)
    out[warning] = 1.0
    out[intervened] = 3.0
    out[normal_safe] = 0.2
    return out


def shield_keep_weights(intervened: np.ndarray, warning: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    shield_w = np.zeros(intervened.shape, dtype=np.float32)
    keep_w = np.zeros(intervened.shape, dtype=np.float32)
    normal_safe = ~(intervened | warning)
    shield_w[intervened] = 1.0
    shield_w[warning] = 0.5
    keep_w[warning] = 0.5
    keep_w[normal_safe] = 1.0
    return shield_w, keep_w


def compute_mode_sample_weight(mode: str, hard15: np.ndarray, intervened: np.ndarray, warning: np.ndarray) -> np.ndarray:
    if mode == "exp1_actor_only":
        return np.where(hard15, 3.0, 1.0).astype(np.float32)
    if mode == "exp2_risk_weighted":
        split_weight = np.where(hard15, 5.0, 1.0).astype(np.float32)
        state_weight = state_weight_array(intervened, warning)
        return split_weight * state_weight
    raise ValueError(f"Unsupported distillation weighting mode: {mode}")


def freeze_actor_only(policy: nn.Module) -> Dict[str, List[str]]:
    trainable = []
    frozen = []
    for name, param in policy.named_parameters():
        can_train = name.startswith("mlp_extractor.policy_net.") or name.startswith("action_net.")
        param.requires_grad = can_train
        if can_train:
            trainable.append(name)
        else:
            frozen.append(name)
    return {"trainable": trainable, "frozen": frozen}


def blank_metric_row() -> Dict[str, str]:
    return {key: "" for key in ALL_METRIC_FIELDS}


def evaluate_split(
    model: PPO,
    dataset,
    cfg: Config,
    model_tag: str,
    split_name: str,
    indices: Sequence[int],
    shield_params: Dict,
    thresholds: Dict,
) -> Dict[str, Dict]:
    model_info = get_model_info_by_tag(model_tag)
    no_shield_rows = [
        {**evaluate_single_trajectory(model, dataset, cfg, traj_index), "model_tag": model_tag}
        for traj_index in indices
    ]
    shield_rows = [
        {
            **evaluate_single_trajectory_with_shield(model, dataset, cfg, traj_index, shield_params),
            "model_tag": model_tag,
        }
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


def evaluate_for_epoch(
    model: PPO,
    dataset,
    cfg: Config,
    model_tag: str,
    shield_params: Dict,
    thresholds: Dict,
    hard_indices: Sequence[int],
    full_indices: Sequence[int],
    include_full: bool,
) -> Dict[str, Dict]:
    result = {
        "hard15": evaluate_split(model, dataset, cfg, model_tag, "hard15", hard_indices, shield_params, thresholds)
    }
    if include_full:
        result["full"] = evaluate_split(model, dataset, cfg, model_tag, "full", full_indices, shield_params, thresholds)
    return result


def make_eval_csv_row(epoch: int, step_label: str, eval_summaries: Dict[str, Dict], include_full: bool) -> Dict:
    row = {"epoch": int(epoch), "step_label": step_label, "include_full": bool(include_full)}
    row.update(blank_metric_row())
    if include_full:
        row.update(flatten_eval_summaries(eval_summaries))
        return row

    hard = eval_summaries["hard15"]
    hard_only = {"hard15": hard, "full": {"no_shield_summary": {}, "shield_summary": {}}}
    row.update(flatten_eval_summaries(hard_only))
    for key in ALL_METRIC_FIELDS:
        if key.startswith("full_"):
            row[key] = ""
    return row


def maybe_update_best(
    model: PPO,
    output_dir: Path,
    epoch: int,
    eval_summaries: Dict[str, Dict],
    best_key: Optional[Tuple[float, float, float, float, float]],
    best_epoch: Optional[int],
) -> Tuple[Optional[Tuple[float, float, float, float, float]], Optional[int], bool]:
    current_key = build_best_key(eval_summaries)
    improved = best_key is None or current_key > best_key
    if improved:
        best_prefix = output_dir / "best_checkpoint"
        model.save(str(best_prefix))
        save_json(
            output_dir / "best_model_info.json",
            {
                "best_epoch": int(epoch),
                "best_checkpoint_path": str(best_prefix.with_suffix(".zip")),
                "best_key": list(current_key),
                "full_shield_summary": eval_summaries["full"]["shield_summary"],
                "full_no_shield_summary": eval_summaries["full"]["no_shield_summary"],
                "hard15_shield_summary": eval_summaries["hard15"]["shield_summary"],
                "hard15_no_shield_summary": eval_summaries["hard15"]["no_shield_summary"],
            },
        )
        return current_key, epoch, True
    return best_key, best_epoch, False


def distill_epoch(
    model: PPO,
    optimizer: Adam,
    teacher_arrays: Dict[str, np.ndarray],
    sample_weight: np.ndarray,
    shield_weight: np.ndarray,
    keep_weight: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    indices = np.arange(len(teacher_arrays["obs"]))
    np.random.shuffle(indices)
    policy = model.policy
    policy.set_training_mode(True)

    loss_sum = 0.0
    shield_loss_sum = 0.0
    keep_loss_sum = 0.0
    batch_count = 0

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        obs_t = torch.as_tensor(teacher_arrays["obs"][batch_idx], device=device)
        policy_action_t = torch.as_tensor(teacher_arrays["policy_actions"][batch_idx], device=device)
        shield_action_t = torch.as_tensor(teacher_arrays["shield_actions"][batch_idx], device=device)
        sample_weight_t = torch.as_tensor(sample_weight[batch_idx], device=device)
        shield_weight_t = torch.as_tensor(shield_weight[batch_idx], device=device)
        keep_weight_t = torch.as_tensor(keep_weight[batch_idx], device=device)

        dist = policy.get_distribution(obs_t)
        mu = dist.distribution.mean
        shield_loss_per = ((mu - shield_action_t) ** 2).mean(dim=1)
        keep_loss_per = ((mu - policy_action_t) ** 2).mean(dim=1)
        per_sample = shield_weight_t * shield_loss_per + keep_weight_t * keep_loss_per
        loss = (sample_weight_t * per_sample).sum() / sample_weight_t.sum().clamp_min(1e-6)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_sum += float(loss.item())
        shield_loss_sum += float((sample_weight_t * shield_loss_per).sum().item() / sample_weight_t.sum().clamp_min(1e-6).item())
        keep_loss_sum += float((sample_weight_t * keep_loss_per).sum().item() / sample_weight_t.sum().clamp_min(1e-6).item())
        batch_count += 1

    return {
        "loss": loss_sum / max(batch_count, 1),
        "shield_loss": shield_loss_sum / max(batch_count, 1),
        "keep_loss": keep_loss_sum / max(batch_count, 1),
    }


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = 0.0
    next_value = last_value
    next_non_terminal = 1.0
    for step in reversed(range(len(rewards))):
        if dones[step]:
            next_non_terminal = 0.0
            next_value = 0.0
        delta = rewards[step] + gamma * next_value * next_non_terminal - values[step]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[step] = last_gae
        next_non_terminal = 1.0 - float(dones[step])
        next_value = values[step]
    returns = advantages + values
    return advantages, returns


def collect_rollout(
    model: PPO,
    env: MergingEnv,
    cfg: Config,
    shield_params: Dict,
    rollout_steps: int,
    device: torch.device,
    reset_seed_base: int,
    hard_filenames: set,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float], np.ndarray]:
    policy = model.policy
    policy.set_training_mode(False)
    obs, _ = env.reset(seed=reset_seed_base)
    reset_counter = 1
    episode_step = 0
    episode_interventions = 0
    consecutive_interventions = 0
    safe_steps = 0

    obs_buf = []
    actions_buf = []
    old_log_prob_buf = []
    values_buf = []
    rewards_buf = []
    dones_buf = []
    shield_actions_buf = []
    shield_intervened_buf = []
    shield_warning_buf = []
    hard15_buf = []

    episode_rewards = []
    current_episode_reward = 0.0
    intervention_count = 0
    warning_count = 0

    for _ in range(rollout_steps):
        obs_t = torch.as_tensor(obs, device=device).float().unsqueeze(0)
        with torch.no_grad():
            actions_t, values_t, log_prob_t = policy.forward(obs_t, deterministic=False)
        action = actions_t.cpu().numpy().reshape(-1)
        action = np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)

        teacher_action, shield_info = shield_action(
            env=env,
            policy_action=action,
            params=shield_params,
            step_count=episode_step,
            shield_interventions=episode_interventions,
            consecutive_interventions=consecutive_interventions,
            safe_steps=safe_steps,
        )
        teacher_action = np.asarray(teacher_action, dtype=np.float32).reshape(-1)
        filename = env.current_traj.get("filename", "")
        is_hard = filename in hard_filenames

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
        if bool(shield_info["intervened"]):
            episode_interventions += 1
            consecutive_interventions += 1
        else:
            consecutive_interventions = 0
        if bool(shield_info.get("warning", False)):
            warning_count += 1

        if (
            not bool(getattr(env, "has_collided_this_episode", False))
            and hasattr(env, "_get_surround_at_t")
        ):
            surr_now = env._get_surround_at_t(env.t)
            px, py, _, vy = [float(x) for x in env.ego_state]
            min_ttc, min_thw = env._compute_min_ttc_thw(px, py, vy, surr_now)
            if (
                min_ttc >= shield_params.get("lead_ttc_min", 3.0)
                and min_thw >= shield_params.get("follow_thw_min", 0.8)
            ):
                safe_steps += 1
            else:
                safe_steps = 0
        else:
            safe_steps = 0

        obs_buf.append(np.asarray(obs, dtype=np.float32).copy())
        actions_buf.append(action.copy())
        old_log_prob_buf.append(float(log_prob_t.cpu().numpy().reshape(-1)[0]))
        values_buf.append(float(values_t.cpu().numpy().reshape(-1)[0]))
        rewards_buf.append(float(reward))
        dones_buf.append(done)
        shield_actions_buf.append(teacher_action.copy())
        shield_intervened_buf.append(bool(shield_info["intervened"]))
        shield_warning_buf.append(bool(shield_info.get("warning", False)))
        hard15_buf.append(bool(is_hard))

        current_episode_reward += float(reward)
        intervention_count += int(bool(shield_info["intervened"]))
        episode_step += 1

        obs = next_obs
        if done:
            episode_rewards.append(current_episode_reward)
            current_episode_reward = 0.0
            obs, _ = env.reset(seed=reset_seed_base + reset_counter)
            reset_counter += 1
            episode_step = 0
            episode_interventions = 0
            consecutive_interventions = 0
            safe_steps = 0

    with torch.no_grad():
        last_value = float(policy.predict_values(torch.as_tensor(obs, device=device).float().unsqueeze(0)).cpu().numpy().reshape(-1)[0])

    rewards_arr = np.asarray(rewards_buf, dtype=np.float32)
    values_arr = np.asarray(values_buf, dtype=np.float32)
    dones_arr = np.asarray(dones_buf, dtype=np.bool_)
    advantages, returns = compute_gae(
        rewards=rewards_arr,
        values=values_arr,
        dones=dones_arr,
        last_value=last_value,
        gamma=float(model.gamma),
        gae_lambda=float(model.gae_lambda),
    )

    payload = {
        "obs": np.asarray(obs_buf, dtype=np.float32),
        "actions": np.asarray(actions_buf, dtype=np.float32),
        "old_log_prob": np.asarray(old_log_prob_buf, dtype=np.float32),
        "old_values": values_arr,
        "returns": returns.astype(np.float32),
        "advantages": advantages.astype(np.float32),
        "shield_actions": np.asarray(shield_actions_buf, dtype=np.float32),
        "shield_intervened": np.asarray(shield_intervened_buf, dtype=np.bool_),
        "shield_warning": np.asarray(shield_warning_buf, dtype=np.bool_),
        "hard15": np.asarray(hard15_buf, dtype=np.bool_),
    }
    stats = {
        "rollout_reward_mean": float(np.mean(episode_rewards)) if episode_rewards else float(np.mean(rewards_arr)),
        "rollout_reward_sum": float(np.sum(rewards_arr)),
        "shield_intervention_rate": float(intervention_count / max(len(rewards_arr), 1)),
        "shield_warning_rate": float(warning_count / max(len(rewards_arr), 1)),
        "episodes_finished": len(episode_rewards),
    }
    return payload, stats, obs


def ppo_finetune_epoch(
    model: PPO,
    optimizer: Adam,
    rollout: Dict[str, np.ndarray],
    batch_size: int,
    update_epochs: int,
    lambda_shield: float,
    outer_epoch: int,
    outer_total_epochs: int,
    device: torch.device,
) -> Dict[str, float]:
    policy = model.policy
    policy.set_training_mode(True)

    obs = torch.as_tensor(rollout["obs"], device=device)
    actions = torch.as_tensor(rollout["actions"], device=device)
    old_log_prob = torch.as_tensor(rollout["old_log_prob"], device=device)
    returns = torch.as_tensor(rollout["returns"], device=device)
    advantages = torch.as_tensor(rollout["advantages"], device=device)
    shield_actions = torch.as_tensor(rollout["shield_actions"], device=device)
    shield_intervened = rollout["shield_intervened"]
    shield_warning = rollout["shield_warning"]
    hard15 = rollout["hard15"]

    sample_weight = compute_mode_sample_weight(
        "exp2_risk_weighted",
        hard15=hard15,
        intervened=shield_intervened,
        warning=shield_warning,
    )
    sample_weight_t = torch.as_tensor(sample_weight, device=device)
    advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-8)

    clip_range = model.clip_range
    progress_remaining = max(0.0, 1.0 - (outer_epoch / max(outer_total_epochs, 1)))
    clip_value = clip_range(progress_remaining) if callable(clip_range) else float(clip_range)

    total_loss = 0.0
    policy_loss_total = 0.0
    value_loss_total = 0.0
    entropy_loss_total = 0.0
    aux_loss_total = 0.0
    batch_counter = 0

    data_size = obs.shape[0]
    for _ in range(update_epochs):
        indices = np.arange(data_size)
        np.random.shuffle(indices)
        for start in range(0, data_size, batch_size):
            batch_idx = indices[start : start + batch_size]
            batch_obs = obs[batch_idx]
            batch_actions = actions[batch_idx]
            batch_old_log_prob = old_log_prob[batch_idx]
            batch_returns = returns[batch_idx]
            batch_advantages = advantages[batch_idx]
            batch_shield_actions = shield_actions[batch_idx]
            batch_weight = sample_weight_t[batch_idx]

            values_pred, log_prob, entropy = policy.evaluate_actions(batch_obs, batch_actions)
            values_pred = values_pred.flatten()
            ratio = torch.exp(log_prob - batch_old_log_prob)
            unclipped = ratio * batch_advantages
            clipped = torch.clamp(ratio, 1.0 - clip_value, 1.0 + clip_value) * batch_advantages
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = ((batch_returns - values_pred) ** 2).mean()
            entropy_loss = -entropy.mean()

            dist = policy.get_distribution(batch_obs)
            mu = dist.distribution.mean
            aux_per = ((mu - batch_shield_actions) ** 2).mean(dim=1)
            aux_loss = (batch_weight * aux_per).sum() / batch_weight.sum().clamp_min(1e-6)

            loss = (
                policy_loss
                + float(model.ent_coef) * entropy_loss
                + float(model.vf_coef) * value_loss
                + float(lambda_shield) * aux_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), float(model.max_grad_norm))
            optimizer.step()

            total_loss += float(loss.item())
            policy_loss_total += float(policy_loss.item())
            value_loss_total += float(value_loss.item())
            entropy_loss_total += float(entropy_loss.item())
            aux_loss_total += float(aux_loss.item())
            batch_counter += 1

    return {
        "loss": total_loss / max(batch_counter, 1),
        "policy_loss": policy_loss_total / max(batch_counter, 1),
        "value_loss": value_loss_total / max(batch_counter, 1),
        "entropy_loss": entropy_loss_total / max(batch_counter, 1),
        "aux_loss": aux_loss_total / max(batch_counter, 1),
    }


def save_config(output_dir: Path, payload: Dict) -> None:
    save_json(output_dir / "config.json", payload)


def run_distillation_mode(
    mode: str,
    args,
    teacher_payload: Dict,
    base_checkpoint: Path,
    output_dir: Path,
    device: torch.device,
) -> None:
    cfg = Config()
    shield_params = build_v10b_shield_params()
    thresholds = {"endpoint": 0.95, "safety": 0.90, "collision": 0.01}

    model = PPO.load(str(base_checkpoint), device=device)
    teacher_arrays = build_teacher_arrays(teacher_payload["samples"])
    sample_weight = compute_mode_sample_weight(
        mode,
        hard15=teacher_arrays["hard15"],
        intervened=teacher_arrays["intervened"],
        warning=teacher_arrays["warning"],
    )
    shield_weight, keep_weight = shield_keep_weights(teacher_arrays["intervened"], teacher_arrays["warning"])
    freeze_summary = freeze_actor_only(model.policy)
    optimizer = Adam(
        [param for param in model.policy.parameters() if param.requires_grad],
        lr=float(args.lr),
    )

    eval_dataset = load_eval_dataset(device=device)
    _, _, hard_indices, _ = load_hard_case_info(
        eval_dataset,
        hard_list_path=teacher_payload["meta"].get("hard_list_path"),
    )
    full_indices = list(range(len(eval_dataset)))
    hard_indices = limit_indices(hard_indices, args.eval_hard_limit or None)
    full_indices = limit_indices(full_indices, args.eval_full_limit or None)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(
        output_dir,
        {
            "mode": mode,
            "base_checkpoint": str(base_checkpoint),
            "seed": int(args.seed),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "teacher_dataset": str(Path(args.teacher_dataset).resolve()),
            "teacher_sample_count": int(len(teacher_payload["samples"])),
            "shield_variant": shield_params["shield_variant"],
            "trainable_params": freeze_summary["trainable"],
            "frozen_param_count": len(freeze_summary["frozen"]),
            "eval_full_every": int(args.eval_full_every),
            "eval_full_limit": int(args.eval_full_limit),
            "eval_hard_limit": int(args.eval_hard_limit),
        },
    )

    best_key = None
    best_epoch = None
    eval_csv = output_dir / "eval_summary.csv"
    loss_csv = output_dir / "loss.csv"

    epoch0_eval = evaluate_for_epoch(
        model=model,
        dataset=eval_dataset,
        cfg=cfg,
        model_tag=BASE_MODEL_TAG,
        shield_params=shield_params,
        thresholds=thresholds,
        hard_indices=hard_indices,
        full_indices=full_indices,
        include_full=True,
    )
    append_csv_row(eval_csv, make_eval_csv_row(0, "epoch0", epoch0_eval, include_full=True))
    best_key, best_epoch, _ = maybe_update_best(model, output_dir, 0, epoch0_eval, best_key, best_epoch)

    for epoch in range(1, int(args.epochs) + 1):
        train_stats = distill_epoch(
            model=model,
            optimizer=optimizer,
            teacher_arrays=teacher_arrays,
            sample_weight=sample_weight,
            shield_weight=shield_weight,
            keep_weight=keep_weight,
            batch_size=int(args.batch_size),
            device=device,
        )
        append_csv_row(
            loss_csv,
            {
                "epoch": epoch,
                "loss": train_stats["loss"],
                "shield_loss": train_stats["shield_loss"],
                "keep_loss": train_stats["keep_loss"],
            },
        )

        include_full = (epoch % int(args.eval_full_every) == 0) or (epoch == int(args.epochs))
        eval_summaries = evaluate_for_epoch(
            model=model,
            dataset=eval_dataset,
            cfg=cfg,
            model_tag=BASE_MODEL_TAG,
            shield_params=shield_params,
            thresholds=thresholds,
            hard_indices=hard_indices,
            full_indices=full_indices,
            include_full=include_full,
        )
        append_csv_row(eval_csv, make_eval_csv_row(epoch, "train_epoch", eval_summaries, include_full=include_full))
        if include_full:
            best_key, best_epoch, improved = maybe_update_best(
                model, output_dir, epoch, eval_summaries, best_key, best_epoch
            )
            if improved:
                print(f"[*] {mode}: best checkpoint updated at epoch {epoch}")

    save_json(
        output_dir / "run_summary.json",
        {
            "mode": mode,
            "best_epoch": best_epoch,
            "best_checkpoint": str((output_dir / "best_checkpoint.zip").resolve()) if best_epoch is not None else "",
        },
    )


def run_ppo_finetune_mode(
    args,
    teacher_payload: Dict,
    checkpoint: Path,
    output_dir: Path,
    device: torch.device,
) -> None:
    cfg = Config()
    shield_params = build_v10b_shield_params()
    thresholds = {"endpoint": 0.95, "safety": 0.90, "collision": 0.01}

    model = PPO.load(str(checkpoint), device=device)
    optimizer = Adam(model.policy.parameters(), lr=float(args.lr))

    eval_dataset = load_eval_dataset(device=device)
    hard_list_path, hard_filenames, hard_indices, _ = load_hard_case_info(
        eval_dataset,
        hard_list_path=teacher_payload["meta"].get("hard_list_path"),
    )
    full_indices = list(range(len(eval_dataset)))
    hard_indices = limit_indices(hard_indices, args.eval_hard_limit or None)
    full_indices = limit_indices(full_indices, args.eval_full_limit or None)

    train_dataset = load_train_dataset(device=device, seed=args.seed)
    train_env = MergingEnv(train_dataset)
    train_env.collision_margin = 1.0

    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(
        output_dir,
        {
            "mode": "exp3_ppo_finetune",
            "base_checkpoint": str(checkpoint),
            "seed": int(args.seed),
            "outer_epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "lambda_shield": float(args.lambda_shield),
            "rollout_steps": int(args.rollout_steps),
            "ppo_update_epochs": int(args.ppo_update_epochs),
            "hard_list_path": str(hard_list_path),
            "hard_filename_count": len(hard_filenames),
            "shield_variant": shield_params["shield_variant"],
        },
    )

    best_key = None
    best_epoch = None
    eval_csv = output_dir / "eval_summary.csv"
    loss_csv = output_dir / "loss.csv"

    epoch0_eval = evaluate_for_epoch(
        model=model,
        dataset=eval_dataset,
        cfg=cfg,
        model_tag=BASE_MODEL_TAG,
        shield_params=shield_params,
        thresholds=thresholds,
        hard_indices=hard_indices,
        full_indices=full_indices,
        include_full=True,
    )
    append_csv_row(eval_csv, make_eval_csv_row(0, "epoch0", epoch0_eval, include_full=True))
    best_key, best_epoch, _ = maybe_update_best(model, output_dir, 0, epoch0_eval, best_key, best_epoch)

    for epoch in range(1, int(args.epochs) + 1):
        rollout, rollout_stats, _ = collect_rollout(
            model=model,
            env=train_env,
            cfg=cfg,
            shield_params=shield_params,
            rollout_steps=int(args.rollout_steps),
            device=device,
            reset_seed_base=int(args.seed + epoch * 1000),
            hard_filenames=set(hard_filenames),
        )
        train_stats = ppo_finetune_epoch(
            model=model,
            optimizer=optimizer,
            rollout=rollout,
            batch_size=int(args.batch_size),
            update_epochs=int(args.ppo_update_epochs),
            lambda_shield=float(args.lambda_shield),
            outer_epoch=epoch,
            outer_total_epochs=int(args.epochs),
            device=device,
        )
        append_csv_row(
            loss_csv,
            {
                "epoch": epoch,
                **rollout_stats,
                **train_stats,
            },
        )

        include_full = (epoch % int(args.eval_full_every) == 0) or (epoch == int(args.epochs))
        eval_summaries = evaluate_for_epoch(
            model=model,
            dataset=eval_dataset,
            cfg=cfg,
            model_tag=BASE_MODEL_TAG,
            shield_params=shield_params,
            thresholds=thresholds,
            hard_indices=hard_indices,
            full_indices=full_indices,
            include_full=include_full,
        )
        append_csv_row(eval_csv, make_eval_csv_row(epoch, "ppo_epoch", eval_summaries, include_full=include_full))
        if include_full:
            best_key, best_epoch, improved = maybe_update_best(
                model, output_dir, epoch, eval_summaries, best_key, best_epoch
            )
            if improved:
                print(f"[*] exp3_ppo_finetune: best checkpoint updated at epoch {epoch}")

    save_json(
        output_dir / "run_summary.json",
        {
            "mode": "exp3_ppo_finetune",
            "best_epoch": best_epoch,
            "best_checkpoint": str((output_dir / "best_checkpoint.zip").resolve()) if best_epoch is not None else "",
        },
    )


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir).resolve() if args.output_dir else default_output_dir(args.mode).resolve()
    teacher_payload = load_teacher_payload(Path(args.teacher_dataset).resolve())
    base_checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else resolve_base_checkpoint_from_teacher(teacher_payload)

    defaults = {
        "exp1_actor_only": {"epochs": 10, "batch_size": 256, "lr": 1e-5},
        "exp2_risk_weighted": {"epochs": 20, "batch_size": 256, "lr": 1e-5},
        "exp3_ppo_finetune": {"epochs": 30, "batch_size": 256, "lr": 1e-5},
    }
    args.epochs = int(args.epochs or defaults[args.mode]["epochs"])
    args.batch_size = int(args.batch_size or defaults[args.mode]["batch_size"])
    args.lr = float(args.lr or defaults[args.mode]["lr"])
    cfg = Config()
    args.rollout_steps = int(args.rollout_steps or cfg.STEPS_PER_EPOCH)

    if args.mode == "exp3_ppo_finetune":
        temp_model = PPO.load(str(base_checkpoint), device=device)
        args.ppo_update_epochs = int(args.ppo_update_epochs or temp_model.n_epochs)
        del temp_model
    else:
        args.ppo_update_epochs = int(args.ppo_update_epochs or 0)

    seed_everything(args.seed, deterministic=True)
    print("=" * 80)
    print(f"Mode: {args.mode}")
    print(f"Teacher dataset: {Path(args.teacher_dataset).resolve()}")
    print(f"Base checkpoint: {base_checkpoint}")
    print(f"Output dir: {output_dir}")
    print(f"Device: {device}")
    print("=" * 80)

    if args.mode in {"exp1_actor_only", "exp2_risk_weighted"}:
        run_distillation_mode(
            mode=args.mode,
            args=args,
            teacher_payload=teacher_payload,
            base_checkpoint=base_checkpoint,
            output_dir=output_dir,
            device=device,
        )
    else:
        run_ppo_finetune_mode(
            args=args,
            teacher_payload=teacher_payload,
            checkpoint=base_checkpoint,
            output_dir=output_dir,
            device=device,
        )


if __name__ == "__main__":
    main()
