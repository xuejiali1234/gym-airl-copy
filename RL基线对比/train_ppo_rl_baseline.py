import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import MlpPolicy


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from configs.config import Config
from envs.merging_env import MergingEnv
from model.attention_net import AttentionFeaturesExtractor, GoalConditionedMLPFeaturesExtractor
from ppo_rl_reward_wrapper import PpoRlRewardWrapper
from train_airl_baseline import (
    append_eval_metrics,
    build_safe_checkpoint_path,
    config_to_dict,
    evaluate_policy_metrics,
    sanitize_run_tag,
    save_run_metadata,
    seed_everything,
    split_dataset,
)
from utils.data_loader import MergingDataset


DATA_PATHS = [
    ROOT_DIR / "data" / "lane_change_trajectories-0750am-0805am",
    ROOT_DIR / "data" / "lane_change_trajectories-0805am-0820am",
    ROOT_DIR / "data" / "lane_change_trajectories-0820am-0835am",
]

PROGRESS_FIELDS = [
    "epoch",
    "total_timesteps",
    "episodes_completed",
    "train_reward_mean",
    "train_reward_std",
    "train_episode_length_mean",
    "train_episode_length_std",
    "term_eff",
    "term_safety",
    "term_thw",
    "term_comfort",
    "term_goal",
    "term_speed_over",
    "term_merge_bonus",
    "term_success_bonus",
    "term_timeout_penalty",
    "term_collision_penalty",
    "term_reward_raw",
    "term_reward_clipped",
    "mean_min_ttc_train",
    "mean_min_thw_train",
    "mean_abs_jerk_x_train",
    "mean_abs_jerk_train",
    "mean_abs_jerk_y_train",
    "mean_comfort_jerk2d_train",
    "mean_speed_train_mps",
    "merge_bonus_rate",
    "success_bonus_rate",
    "timeout_penalty_rate",
    "collision_penalty_rate",
    "rollout_ep_rew_mean",
    "rollout_ep_len_mean",
    "train_approx_kl",
    "train_value_loss",
    "train_policy_gradient_loss",
    "train_entropy_loss",
    "train_clip_fraction",
    "ppo_rl_lr",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Train the standalone PPO-RL handcrafted-reward baseline.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--tag", default="")
    parser.add_argument("--goal", type=int, choices=[0, 1], default=None)
    parser.add_argument("--attention", type=int, choices=[0, 1], default=None)
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def parse_env_bool(name):
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean environment value for {name}: {value}")


def add_ppo_rl_defaults(cfg):
    defaults = {
        "PPO_RL_LR": 8e-5,
        "PPO_RL_GAMMA": 0.99,
        "PPO_RL_GAE_LAMBDA": 0.95,
        "PPO_RL_CLIP_RANGE": 0.2,
        "PPO_RL_VF_COEF": 0.5,
        "PPO_RL_MAX_GRAD_NORM": 0.5,
        "PPO_RL_REWARD_CLIP_MIN": -10.0,
        "PPO_RL_REWARD_CLIP_MAX": 3.0,
        "PPO_RL_JERK_NORM": 3.0,
        "PPO_RL_JERK_X_NORM": 10.0,
        "PPO_RL_JERK_Y_NORM": 10.0,
        "PPO_RL_EFF_MODE": "speed_limit",
        "PPO_RL_TARGET_SPEED_MPS": 15.0,
        "PPO_RL_TARGET_SPEED_BAND_MPS": 5.0,
        "PPO_RL_SPEED_OVER_MPS": 17.0,
        "PPO_RL_SPEED_OVER_BAND_MPS": 5.0,
        "PPO_RL_GOAL_PROGRESS_SCALE": 20.0,
        "PPO_RL_THW_SAFE_SECONDS": 2.0,
        "PPO_RL_USE_GOAL": True,
        "PPO_RL_USE_ATTENTION": False,
        "PPO_RL_W_EFF": 0.20,
        "PPO_RL_W_SAFETY": 1.00,
        "PPO_RL_W_THW": 0.0,
        "PPO_RL_W_COMFORT": 0.05,
        "PPO_RL_W_GOAL": 0.80,
        "PPO_RL_W_SPEED_OVER": 0.0,
        "PPO_RL_COLLISION_PENALTY": -5.0,
        "PPO_RL_SUCCESS_BONUS": 1.0,
        "PPO_RL_MERGE_BONUS": 0.5,
        "PPO_RL_TIMEOUT_PENALTY": -1.0,
        "PPO_RL_SAVE_FREQ_EPOCHS": 1,
        "PPO_RL_QUICK_EVAL_EPISODES": 8,
        "PPO_RL_FULL_EVAL_EPISODES": 100,
        "PPO_RL_FULL_EVAL_FREQ_EPOCHS": 1,
        "PPO_RL_EPOCH0_EVAL_EPISODES": 100,
        "PPO_RL_BEST_SELECT_START_EPOCH": 270,
    }
    for key, value in defaults.items():
        setattr(cfg, key, value)


def apply_env_overrides(cfg):
    overrides = {}

    def set_if_present(env_name, attr_name, caster):
        raw = os.environ.get(env_name)
        if raw is None or raw == "":
            return
        value = caster(raw)
        setattr(cfg, attr_name, value)
        overrides[attr_name] = value

    for env_name, attr_name, caster in (
        ("PPO_RL_EPOCHS", "EPOCHS", int),
        ("PPO_RL_SEED", "SEED", int),
        ("PPO_RL_STEPS_PER_EPOCH", "STEPS_PER_EPOCH", int),
        ("PPO_RL_PPO_EPOCHS", "PPO_EPOCHS", int),
        ("PPO_RL_PPO_MINI_BATCH_SIZE", "PPO_MINI_BATCH_SIZE", int),
        ("PPO_RL_ENT_COEF", "PPO_ENT_COEF", float),
        ("PPO_RL_LR", "PPO_RL_LR", float),
        ("PPO_RL_GAMMA", "PPO_RL_GAMMA", float),
        ("PPO_RL_GAE_LAMBDA", "PPO_RL_GAE_LAMBDA", float),
        ("PPO_RL_CLIP_RANGE", "PPO_RL_CLIP_RANGE", float),
        ("PPO_RL_VF_COEF", "PPO_RL_VF_COEF", float),
        ("PPO_RL_MAX_GRAD_NORM", "PPO_RL_MAX_GRAD_NORM", float),
        ("PPO_RL_W_EFF", "PPO_RL_W_EFF", float),
        ("PPO_RL_W_SAFETY", "PPO_RL_W_SAFETY", float),
        ("PPO_RL_W_THW", "PPO_RL_W_THW", float),
        ("PPO_RL_W_COMFORT", "PPO_RL_W_COMFORT", float),
        ("PPO_RL_W_GOAL", "PPO_RL_W_GOAL", float),
        ("PPO_RL_W_SPEED_OVER", "PPO_RL_W_SPEED_OVER", float),
        ("PPO_RL_COLLISION_PENALTY", "PPO_RL_COLLISION_PENALTY", float),
        ("PPO_RL_SUCCESS_BONUS", "PPO_RL_SUCCESS_BONUS", float),
        ("PPO_RL_MERGE_BONUS", "PPO_RL_MERGE_BONUS", float),
        ("PPO_RL_TIMEOUT_PENALTY", "PPO_RL_TIMEOUT_PENALTY", float),
        ("PPO_RL_REWARD_CLIP_MIN", "PPO_RL_REWARD_CLIP_MIN", float),
        ("PPO_RL_REWARD_CLIP_MAX", "PPO_RL_REWARD_CLIP_MAX", float),
        ("PPO_RL_JERK_X_NORM", "PPO_RL_JERK_X_NORM", float),
        ("PPO_RL_JERK_Y_NORM", "PPO_RL_JERK_Y_NORM", float),
        ("PPO_RL_TARGET_SPEED_MPS", "PPO_RL_TARGET_SPEED_MPS", float),
        ("PPO_RL_TARGET_SPEED_BAND_MPS", "PPO_RL_TARGET_SPEED_BAND_MPS", float),
        ("PPO_RL_SPEED_OVER_MPS", "PPO_RL_SPEED_OVER_MPS", float),
        ("PPO_RL_SPEED_OVER_BAND_MPS", "PPO_RL_SPEED_OVER_BAND_MPS", float),
        ("PPO_RL_GOAL_PROGRESS_SCALE", "PPO_RL_GOAL_PROGRESS_SCALE", float),
        ("PPO_RL_THW_SAFE_SECONDS", "PPO_RL_THW_SAFE_SECONDS", float),
        ("PPO_RL_SAVE_FREQ_EPOCHS", "PPO_RL_SAVE_FREQ_EPOCHS", int),
        ("PPO_RL_QUICK_EVAL_EPISODES", "PPO_RL_QUICK_EVAL_EPISODES", int),
        ("PPO_RL_FULL_EVAL_EPISODES", "PPO_RL_FULL_EVAL_EPISODES", int),
        ("PPO_RL_FULL_EVAL_FREQ_EPOCHS", "PPO_RL_FULL_EVAL_FREQ_EPOCHS", int),
        ("PPO_RL_EPOCH0_EVAL_EPISODES", "PPO_RL_EPOCH0_EVAL_EPISODES", int),
        ("PPO_RL_BEST_SELECT_START_EPOCH", "PPO_RL_BEST_SELECT_START_EPOCH", int),
    ):
        set_if_present(env_name, attr_name, caster)

    eff_mode = os.environ.get("PPO_RL_EFF_MODE")
    if eff_mode is not None and eff_mode.strip():
        value = eff_mode.strip().lower()
        if value not in {"speed_limit", "target_speed"}:
            raise ValueError(
                "PPO_RL_EFF_MODE must be 'speed_limit' or 'target_speed', "
                f"got {eff_mode!r}"
            )
        cfg.PPO_RL_EFF_MODE = value
        overrides["PPO_RL_EFF_MODE"] = value

    for env_name, attr_name in (
        ("PPO_RL_USE_GOAL", "PPO_RL_USE_GOAL"),
        ("PPO_RL_USE_ATTENTION", "PPO_RL_USE_ATTENTION"),
    ):
        value = parse_env_bool(env_name)
        if value is not None:
            setattr(cfg, attr_name, value)
            overrides[attr_name] = value

    return overrides


def apply_arg_overrides(cfg, args):
    overrides = {}
    if args.epochs is not None:
        cfg.EPOCHS = int(args.epochs)
        overrides["EPOCHS"] = cfg.EPOCHS
    if args.seed is not None:
        cfg.SEED = int(args.seed)
        overrides["SEED"] = cfg.SEED
    if args.goal is not None:
        cfg.PPO_RL_USE_GOAL = bool(args.goal)
        overrides["PPO_RL_USE_GOAL"] = cfg.PPO_RL_USE_GOAL
    if args.attention is not None:
        cfg.PPO_RL_USE_ATTENTION = bool(args.attention)
        overrides["PPO_RL_USE_ATTENTION"] = cfg.PPO_RL_USE_ATTENTION
    return overrides


def finalize_cfg(cfg):
    cfg.ENABLE_GOAL_CONDITION = bool(getattr(cfg, "PPO_RL_USE_GOAL", True))
    cfg.GOAL_ABLATION_MODE = "normal"
    cfg.ENABLE_ATTENTION = bool(getattr(cfg, "PPO_RL_USE_ATTENTION", False))
    cfg.ATTENTION_ABLATION_MODE = "normal"

    # Explicitly keep AIRL-side safety/reward shaping off for this standalone RL baseline.
    cfg.ENABLE_SAFETY_MODULE = False
    cfg.ENABLE_SAFETY_BRANCH = False
    cfg.ENABLE_SAFETY_AUX_LOSS = False
    cfg.ENABLE_PREDICTIVE_SAFETY_CRITIC = False
    cfg.ENABLE_PREDICTIVE_SAFETY_RESIDUAL = False
    cfg.PREDICTIVE_SAFETY_ENABLE_CPAIR_ADDITIVE = False


def build_run_label(cfg, tag_override):
    if tag_override:
        tag = sanitize_run_tag(tag_override)
    else:
        goal_name = "Goal" if cfg.ENABLE_GOAL_CONDITION else "NoGoal"
        attn_name = "Attn" if cfg.ENABLE_ATTENTION else "NoAttn"
        tag = f"PPO_RL_{goal_name}_{attn_name}_S{cfg.SEED}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return tag, f"ppo_rl_baseline_{tag}_{timestamp}"


def make_log_dir(run_label, output_dir):
    if output_dir:
        log_dir = Path(output_dir).expanduser().resolve()
    else:
        log_dir = ROOT_DIR / "train_log" / run_label
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return log_dir


def append_progress_row(log_dir, row):
    progress_path = Path(log_dir) / "progress.csv"
    file_exists = progress_path.exists()
    with progress_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PROGRESS_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in PROGRESS_FIELDS})


def unwrap_reward_wrapper(vec_env):
    env = vec_env.envs[0]
    while env is not None:
        if isinstance(env, PpoRlRewardWrapper):
            return env
        env = getattr(env, "env", None)
    raise RuntimeError("Failed to locate PpoRlRewardWrapper inside the vec env.")


def mean_and_std(values):
    if not values:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=np.float32)
    return float(np.mean(arr)), float(np.std(arr))


def summarize_completed_episodes(completed):
    summary = {"episodes_completed": float(len(completed))}
    if not completed:
        for key in PROGRESS_FIELDS:
            summary.setdefault(key, float("nan"))
        summary["episodes_completed"] = 0.0
        return summary

    def avg(name):
        return float(np.mean([item.get(name, 0.0) for item in completed]))

    reward_mean, reward_std = mean_and_std([item["train_reward_mean"] for item in completed])
    length_mean, length_std = mean_and_std([item["train_episode_length_mean"] for item in completed])

    summary.update(
        {
            "train_reward_mean": reward_mean,
            "train_reward_std": reward_std,
            "train_episode_length_mean": length_mean,
            "train_episode_length_std": length_std,
            "term_eff": avg("term_eff"),
            "term_safety": avg("term_safety"),
            "term_thw": avg("term_thw"),
            "term_comfort": avg("term_comfort"),
            "term_goal": avg("term_goal"),
            "term_speed_over": avg("term_speed_over"),
            "term_merge_bonus": avg("term_merge_bonus"),
            "term_success_bonus": avg("term_success_bonus"),
            "term_timeout_penalty": avg("term_timeout_penalty"),
            "term_collision_penalty": avg("term_collision_penalty"),
            "term_reward_raw": avg("term_reward_raw"),
            "term_reward_clipped": avg("term_reward_clipped"),
            "mean_min_ttc_train": avg("mean_min_ttc_train"),
            "mean_min_thw_train": avg("mean_min_thw_train"),
            "mean_abs_jerk_x_train": avg("mean_abs_jerk_x_train"),
            "mean_abs_jerk_train": avg("mean_abs_jerk_train"),
            "mean_abs_jerk_y_train": avg("mean_abs_jerk_y_train"),
            "mean_comfort_jerk2d_train": avg("mean_comfort_jerk2d_train"),
            "mean_speed_train_mps": avg("mean_speed_train_mps"),
            "merge_bonus_rate": avg("merge_bonus_rate"),
            "success_bonus_rate": avg("success_bonus_rate"),
            "timeout_penalty_rate": avg("timeout_penalty_rate"),
            "collision_penalty_rate": avg("collision_penalty_rate"),
        }
    )
    return summary


def safe_logger_metric(model, key):
    value = model.logger.name_to_value.get(key)
    if value is None:
        if key == "rollout/ep_rew_mean":
            buffer = list(getattr(model, "ep_info_buffer", []))
            if buffer:
                return float(np.mean([item.get("r", 0.0) for item in buffer]))
        if key == "rollout/ep_len_mean":
            buffer = list(getattr(model, "ep_info_buffer", []))
            if buffer:
                return float(np.mean([item.get("l", 0.0) for item in buffer]))
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def build_selection_score(metrics):
    return (
        float(metrics.get("endpoint_success_rate", 0.0))
        + float(metrics.get("merge_success_rate", 0.0))
        + float(metrics.get("safety_success_rate", 0.0))
        - 2.0 * float(metrics.get("collision_rate", 1.0))
        - 0.02 * float(metrics.get("mean_abs_jerk", 0.0))
    )


def build_policy_kwargs(cfg):
    if cfg.ENABLE_ATTENTION:
        policy_kwargs = dict(
            features_extractor_class=AttentionFeaturesExtractor,
            features_extractor_kwargs=dict(hidden_dim=64, attention_ablation_mode="normal"),
            net_arch=dict(pi=[128, 128], vf=[128, 128]),
            activation_fn=nn.Tanh,
        )
    else:
        policy_kwargs = dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128]),
            activation_fn=nn.Tanh,
        )
        if cfg.ENABLE_GOAL_CONDITION:
            policy_kwargs.update(
                dict(
                    features_extractor_class=GoalConditionedMLPFeaturesExtractor,
                    features_extractor_kwargs=dict(
                        state_dim=16,
                        goal_dim=2,
                        state_hidden_dim=64,
                        goal_hidden_dim=32,
                    ),
                )
            )
    return policy_kwargs


def main():
    args = parse_args()
    cfg = Config()
    add_ppo_rl_defaults(cfg)
    env_overrides = apply_env_overrides(cfg)
    arg_overrides = apply_arg_overrides(cfg, args)
    finalize_cfg(cfg)

    seed_everything(cfg.SEED, deterministic=getattr(cfg, "DETERMINISTIC_TRAINING", True))
    device = "cuda" if torch.cuda.is_available() and str(getattr(cfg, "DEVICE", "cpu")).startswith("cuda") else "cpu"

    run_tag, run_label = build_run_label(cfg, args.tag or os.environ.get("PPO_RL_TAG", ""))
    log_dir = make_log_dir(run_label, args.output_dir or os.environ.get("PPO_RL_OUTPUT_DIR", ""))
    checkpoint_dir = log_dir / "checkpoints"

    print("=" * 80)
    print("PPO-RL handcrafted baseline")
    print(f"Run tag: {run_tag}")
    print(f"Log dir: {log_dir}")
    print(f"Device: {device}")
    print(
        f"Goal={cfg.ENABLE_GOAL_CONDITION}, Attention={cfg.ENABLE_ATTENTION}, "
        f"epochs={cfg.EPOCHS}, steps/epoch={cfg.STEPS_PER_EPOCH}, seed={cfg.SEED}"
    )
    print("=" * 80)

    data_paths = [str(path) for path in DATA_PATHS]
    dataset = MergingDataset(data_paths, device=device)
    if len(dataset) == 0:
        raise RuntimeError("No trajectories found for PPO-RL baseline training.")
    train_dataset, val_dataset = split_dataset(dataset, train_ratio=0.8, seed=cfg.SEED)
    print(f"[*] Dataset split: train={len(train_dataset)} | val={len(val_dataset)}")

    def make_train_env():
        raw_env = MergingEnv(train_dataset, cfg)
        reward_env = PpoRlRewardWrapper(raw_env, cfg)
        return Monitor(
            reward_env,
            info_keywords=("is_success", "is_merge_success", "is_endpoint_success", "is_safety_success"),
        )

    env = DummyVecEnv([make_train_env])
    env.seed(cfg.SEED)
    env_obs_dim = int(env.observation_space.shape[0])
    expected_obs_dim = 18 if cfg.ENABLE_GOAL_CONDITION else 16
    if env_obs_dim != expected_obs_dim:
        raise ValueError(f"Unexpected observation dim: env={env_obs_dim}, expected={expected_obs_dim}")

    policy_kwargs = build_policy_kwargs(cfg)
    policy_extractor_cls = policy_kwargs.get("features_extractor_class")
    policy_features_extractor_class = (
        policy_extractor_cls.__name__ if policy_extractor_cls is not None else "FlattenExtractor"
    )

    model = PPO(
        env=env,
        policy=MlpPolicy,
        learning_rate=cfg.PPO_RL_LR,
        n_steps=cfg.STEPS_PER_EPOCH,
        batch_size=cfg.PPO_MINI_BATCH_SIZE,
        n_epochs=cfg.PPO_EPOCHS,
        gamma=cfg.PPO_RL_GAMMA,
        gae_lambda=cfg.PPO_RL_GAE_LAMBDA,
        clip_range=cfg.PPO_RL_CLIP_RANGE,
        ent_coef=cfg.PPO_ENT_COEF,
        vf_coef=cfg.PPO_RL_VF_COEF,
        max_grad_norm=cfg.PPO_RL_MAX_GRAD_NORM,
        target_kl=0.01,
        policy_kwargs=policy_kwargs,
        seed=cfg.SEED,
        device=device,
        verbose=1,
    )

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).name),
        "run_tag": run_tag,
        "run_label": run_label,
        "seed": int(cfg.SEED),
        "device": device,
        "data_paths": data_paths,
        "dataset_train_size": len(train_dataset),
        "dataset_val_size": len(val_dataset),
        "observation_dim": env_obs_dim,
        "policy_features_extractor_class": policy_features_extractor_class,
        "effective_goal_mode": "normal" if cfg.ENABLE_GOAL_CONDITION else "disabled",
        "effective_attention_mode": "normal" if cfg.ENABLE_ATTENTION else "disabled",
        "ppo_rl_use_goal": bool(cfg.ENABLE_GOAL_CONDITION),
        "ppo_rl_use_attention": bool(cfg.ENABLE_ATTENTION),
        "reward_weights": {
            "W_EFF": cfg.PPO_RL_W_EFF,
            "W_SAFETY": cfg.PPO_RL_W_SAFETY,
            "W_THW": cfg.PPO_RL_W_THW,
            "W_COMFORT": cfg.PPO_RL_W_COMFORT,
            "W_GOAL": cfg.PPO_RL_W_GOAL,
            "W_SPEED_OVER": cfg.PPO_RL_W_SPEED_OVER,
            "EFF_MODE": cfg.PPO_RL_EFF_MODE,
            "TARGET_SPEED_MPS": cfg.PPO_RL_TARGET_SPEED_MPS,
            "TARGET_SPEED_BAND_MPS": cfg.PPO_RL_TARGET_SPEED_BAND_MPS,
            "SPEED_OVER_MPS": cfg.PPO_RL_SPEED_OVER_MPS,
            "SPEED_OVER_BAND_MPS": cfg.PPO_RL_SPEED_OVER_BAND_MPS,
            "GOAL_PROGRESS_SCALE": cfg.PPO_RL_GOAL_PROGRESS_SCALE,
            "THW_SAFE_SECONDS": cfg.PPO_RL_THW_SAFE_SECONDS,
            "COLLISION_PENALTY": cfg.PPO_RL_COLLISION_PENALTY,
            "SUCCESS_BONUS": cfg.PPO_RL_SUCCESS_BONUS,
            "MERGE_BONUS": cfg.PPO_RL_MERGE_BONUS,
            "TIMEOUT_PENALTY": cfg.PPO_RL_TIMEOUT_PENALTY,
            "REWARD_CLIP_MIN": cfg.PPO_RL_REWARD_CLIP_MIN,
            "REWARD_CLIP_MAX": cfg.PPO_RL_REWARD_CLIP_MAX,
        },
        "env_overrides": env_overrides,
        "arg_overrides": arg_overrides,
        "config": config_to_dict(cfg),
    }
    save_run_metadata(str(log_dir), metadata)

    reward_wrapper = unwrap_reward_wrapper(env)
    epoch0_eval_eps = int(cfg.PPO_RL_EPOCH0_EVAL_EPISODES)
    epoch0_metrics = evaluate_policy_metrics(model, val_dataset, cfg, n_eval_episodes=epoch0_eval_eps)
    append_eval_metrics(
        str(log_dir),
        {
            "epoch": 0,
            "total_timesteps": model.num_timesteps,
            "eval_n_episodes": epoch0_eval_eps,
            "ppo_rl_lr": cfg.PPO_RL_LR,
            "selection_score": build_selection_score(epoch0_metrics),
            **epoch0_metrics,
        },
    )
    print(
        "[*] Epoch 0 eval | "
        f"episodes={epoch0_eval_eps}, "
        f"merge={epoch0_metrics['merge_success_rate']:.3f}, "
        f"endpoint={epoch0_metrics['endpoint_success_rate']:.3f}, "
        f"safety={epoch0_metrics['safety_success_rate']:.3f}, "
        f"collision={epoch0_metrics['collision_rate']:.3f}"
    )

    best_overall_score = float("-inf")
    best_overall_epoch = None
    best_overall_path = None
    best_late_score = float("-inf")
    best_late_epoch = None
    best_late_path = None

    for epoch in range(1, int(cfg.EPOCHS) + 1):
        model.learn(total_timesteps=int(cfg.STEPS_PER_EPOCH), reset_num_timesteps=False)
        completed = reward_wrapper.pop_completed_episode_summaries()
        progress_summary = summarize_completed_episodes(completed)
        progress_row = {
            "epoch": epoch,
            "total_timesteps": model.num_timesteps,
            "episodes_completed": progress_summary.get("episodes_completed", 0.0),
            "train_reward_mean": progress_summary.get("train_reward_mean", float("nan")),
            "train_reward_std": progress_summary.get("train_reward_std", float("nan")),
            "train_episode_length_mean": progress_summary.get("train_episode_length_mean", float("nan")),
            "train_episode_length_std": progress_summary.get("train_episode_length_std", float("nan")),
            "term_eff": progress_summary.get("term_eff", float("nan")),
            "term_safety": progress_summary.get("term_safety", float("nan")),
            "term_thw": progress_summary.get("term_thw", float("nan")),
            "term_comfort": progress_summary.get("term_comfort", float("nan")),
            "term_goal": progress_summary.get("term_goal", float("nan")),
            "term_speed_over": progress_summary.get("term_speed_over", float("nan")),
            "term_merge_bonus": progress_summary.get("term_merge_bonus", float("nan")),
            "term_success_bonus": progress_summary.get("term_success_bonus", float("nan")),
            "term_timeout_penalty": progress_summary.get("term_timeout_penalty", float("nan")),
            "term_collision_penalty": progress_summary.get("term_collision_penalty", float("nan")),
            "term_reward_raw": progress_summary.get("term_reward_raw", float("nan")),
            "term_reward_clipped": progress_summary.get("term_reward_clipped", float("nan")),
            "mean_min_ttc_train": progress_summary.get("mean_min_ttc_train", float("nan")),
            "mean_min_thw_train": progress_summary.get("mean_min_thw_train", float("nan")),
            "mean_abs_jerk_x_train": progress_summary.get("mean_abs_jerk_x_train", float("nan")),
            "mean_abs_jerk_train": progress_summary.get("mean_abs_jerk_train", float("nan")),
            "mean_abs_jerk_y_train": progress_summary.get("mean_abs_jerk_y_train", float("nan")),
            "mean_comfort_jerk2d_train": progress_summary.get("mean_comfort_jerk2d_train", float("nan")),
            "mean_speed_train_mps": progress_summary.get("mean_speed_train_mps", float("nan")),
            "merge_bonus_rate": progress_summary.get("merge_bonus_rate", float("nan")),
            "success_bonus_rate": progress_summary.get("success_bonus_rate", float("nan")),
            "timeout_penalty_rate": progress_summary.get("timeout_penalty_rate", float("nan")),
            "collision_penalty_rate": progress_summary.get("collision_penalty_rate", float("nan")),
            "rollout_ep_rew_mean": safe_logger_metric(model, "rollout/ep_rew_mean"),
            "rollout_ep_len_mean": safe_logger_metric(model, "rollout/ep_len_mean"),
            "train_approx_kl": safe_logger_metric(model, "train/approx_kl"),
            "train_value_loss": safe_logger_metric(model, "train/value_loss"),
            "train_policy_gradient_loss": safe_logger_metric(model, "train/policy_gradient_loss"),
            "train_entropy_loss": safe_logger_metric(model, "train/entropy_loss"),
            "train_clip_fraction": safe_logger_metric(model, "train/clip_fraction"),
            "ppo_rl_lr": cfg.PPO_RL_LR,
        }
        append_progress_row(log_dir, progress_row)

        eval_eps = (
            int(cfg.PPO_RL_FULL_EVAL_EPISODES)
            if epoch % int(cfg.PPO_RL_FULL_EVAL_FREQ_EPOCHS) == 0
            else int(cfg.PPO_RL_QUICK_EVAL_EPISODES)
        )
        eval_metrics = evaluate_policy_metrics(model, val_dataset, cfg, n_eval_episodes=eval_eps)
        selection_score = build_selection_score(eval_metrics)
        append_eval_metrics(
            str(log_dir),
            {
                "epoch": epoch,
                "total_timesteps": model.num_timesteps,
                "eval_n_episodes": eval_eps,
                "ppo_rl_lr": cfg.PPO_RL_LR,
                "selection_score": selection_score,
                **eval_metrics,
            },
        )

        if epoch % int(cfg.PPO_RL_SAVE_FREQ_EPOCHS) == 0:
            checkpoint_path = build_safe_checkpoint_path(
                str(checkpoint_dir),
                run_label,
                probe_tag=run_tag,
                prefix="ppo_rl_policy",
                suffix="epoch",
                epoch=epoch,
            )
            model.save(checkpoint_path)

        if selection_score > best_overall_score:
            best_overall_score = selection_score
            best_overall_epoch = epoch
            best_overall_path = build_safe_checkpoint_path(
                str(checkpoint_dir),
                run_label,
                probe_tag=run_tag,
                prefix="ppo_rl_policy",
                suffix="best_over_training",
                epoch=epoch,
            )
            model.save(best_overall_path)

        if epoch >= int(cfg.PPO_RL_BEST_SELECT_START_EPOCH) and selection_score > best_late_score:
            best_late_score = selection_score
            best_late_epoch = epoch
            best_late_path = build_safe_checkpoint_path(
                str(checkpoint_dir),
                run_label,
                probe_tag=run_tag,
                prefix="ppo_rl_policy",
                suffix="best_after_270",
                epoch=epoch,
            )
            model.save(best_late_path)

        print(
            "[*] Epoch eval | "
            f"epoch={epoch}/{cfg.EPOCHS}, episodes={eval_eps}, "
            f"merge={eval_metrics['merge_success_rate']:.3f}, "
            f"endpoint={eval_metrics['endpoint_success_rate']:.3f}, "
            f"safety={eval_metrics['safety_success_rate']:.3f}, "
            f"collision={eval_metrics['collision_rate']:.3f}, "
            f"dense_norm100={eval_metrics['eval_dense_return_norm100']:.2f}, "
            f"paper_rank={eval_metrics['paper_rank_score_mean']:.2f}, "
            f"selection_score={selection_score:.4f}"
        )

    final_checkpoint_path = build_safe_checkpoint_path(
        str(checkpoint_dir),
        run_label,
        probe_tag=run_tag,
        prefix="ppo_rl_policy",
        suffix="final",
    )
    model.save(final_checkpoint_path)

    summary = {
        "best_overall_epoch": best_overall_epoch,
        "best_overall_score": best_overall_score,
        "best_overall_path": best_overall_path,
        "best_after_270_epoch": best_late_epoch,
        "best_after_270_score": best_late_score,
        "best_after_270_path": best_late_path,
        "final_checkpoint_path": final_checkpoint_path,
    }
    (Path(log_dir) / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n[*] PPO-RL baseline training finished.")
    print(f"[*] Final checkpoint: {final_checkpoint_path}")
    if best_overall_path is not None:
        print(f"[*] Best over training: epoch={best_overall_epoch}, path={best_overall_path}")
    if best_late_path is not None:
        print(f"[*] Best after 270: epoch={best_late_epoch}, path={best_late_path}")


if __name__ == "__main__":
    main()
