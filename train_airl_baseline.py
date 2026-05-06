import os
import csv
import copy
import json
import random
import hashlib
from datetime import datetime
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import sys
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# 导入 imitation 库的核心模块
from imitation.algorithms.adversarial.airl import AIRL
from imitation.data.types import Trajectory
from imitation.rewards.reward_nets import BasicRewardNet, NormalizedRewardNet
from imitation.util.networks import RunningNorm

# 导入 imitation 的原生日志系统
from imitation.util.logger import configure as init_logger 

# 根据你整理的工程结构导入
from utils.data_loader import MergingDataset
from envs.merging_env import MergingEnv
from configs.config import Config

# 导入新建立的注意力网络模块
from model.attention_net import (
    AttentionFeaturesExtractor,
    AttentionRewardNet,
    GoalConditionedMLPFeaturesExtractor,
    GoalRewardWrapper,
)
from model.safety_q_module import SafetyQNetwork, ZeroSafetyQNetwork, SafeQAttentionRewardNet, SafeQMLPRewardNet
from model.predictive_safety_cost import PredictiveSafetyCostNetwork
from model.predictive_safety_oracle import PredictiveSafetyOracle
from model.safety_oracle_q import SafetyOracleQ
from model.safety_pretrain_q import pretrain_safety_q_network, pretrain_predictive_safety_network
from model.safety_airl import MildSafetyAIRL


def seed_everything(seed, deterministic=True):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def config_to_dict(cfg):
    result = {}
    for key in dir(cfg):
        if not key.isupper():
            continue
        value = getattr(cfg, key)
        if callable(value):
            continue
        result[key] = value
    return result


def save_run_metadata(log_dir, payload):
    metadata_path = os.path.join(log_dir, "run_config.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)


def build_safe_checkpoint_path(checkpoint_dir, run_label, *, probe_tag=None, prefix="baseline_policy", suffix="final", epoch=None):
    """Build a checkpoint path that stays safely below Windows MAX_PATH.

    Probe runs already have long directory names; using the full run_label again in
    the checkpoint filename can push the absolute path to ~260 chars and trigger a
    save recursion bug inside SB3 on Windows. Prefer the shorter probe tag for file
    names, and fall back to a compact hashed label if the full path is still long.
    """

    def _normalize_label(value: str) -> str:
        return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)

    def _filename(label: str) -> str:
        if epoch is None:
            return f"{prefix}_{label}_{suffix}.zip"
        return f"{prefix}_{label}_{suffix}_{epoch}.zip"

    file_label = _normalize_label(str(probe_tag or run_label))
    candidate = os.path.join(checkpoint_dir, _filename(file_label))
    if len(os.path.abspath(candidate)) < 245:
        return candidate

    digest = hashlib.md5(str(run_label).encode("utf-8")).hexdigest()[:8]
    compact_label = file_label[:32]
    compact = f"{compact_label}_{digest}"
    candidate = os.path.join(checkpoint_dir, _filename(compact))
    if len(os.path.abspath(candidate)) < 245:
        return candidate

    if epoch is None:
        fallback_name = f"{prefix}_{digest}_{suffix}.zip"
    else:
        fallback_name = f"{prefix}_{digest}_{suffix}_{epoch}.zip"
    return os.path.join(checkpoint_dir, fallback_name)

def convert_to_trajectories(dataset, cfg):
    """将专家数据打包，并根据 Config 决定是否将 goal 拼接到 state 中"""
    trajectories = []
    enable_goal = getattr(cfg, 'ENABLE_GOAL_CONDITION', False)
    
    for traj in dataset.trajectories:
        obs = traj['state']
        acts = traj['action']
        
        # 处理 State: 补齐最后一步
        final_next_state = traj['next_state'][-1].reshape(1, -1)
        full_obs = np.concatenate([obs, final_next_state], axis=0)
        
        # 如果开启 Goal，处理 Goal 并拼接到 State 后面
        if enable_goal:
            obs_goal = traj['goal']
            final_goal = obs_goal[-1].reshape(1, -1)
            full_obs_goal = np.concatenate([obs_goal, final_goal], axis=0)
            
            # 将 [N, 16] 和 [N, 2] 沿特征维度拼接成 [N, 18]
            full_obs = np.concatenate([full_obs, full_obs_goal], axis=-1)
            
        trajectories.append(Trajectory(obs=full_obs, acts=acts, infos=None, terminal=True))
    return trajectories


def split_dataset(dataset, train_ratio=0.8, seed=42):
    """Create train/validation dataset views without reloading files."""
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
        subset.confidence_weights = np.asarray(dataset.confidence_weights)[idxs].copy()
        return subset

    return make_subset(train_indices), make_subset(val_indices)


def evaluate_policy_metrics(model, dataset, cfg, n_eval_episodes=20):
    """Runs policy evaluation without changing the training reward path."""
    eval_env = MergingEnv(dataset)
    merge_successes = []
    endpoint_successes = []
    safety_successes = []
    collision_flags = []
    episode_rewards = []
    dense_episode_rewards = []
    paper_scores = []
    episode_lengths = []
    mean_speeds = []
    max_speeds = []
    mean_abs_accs = []
    mean_abs_jerks = []
    min_ttcs = []
    min_thws = []

    for ep_idx in range(n_eval_episodes):
        obs, _ = eval_env.reset(seed=cfg.SEED + ep_idx)
        terminated = False
        truncated = False
        ep_reward = 0.0
        dense_ep_reward = 0.0
        ep_len = 0
        info = {}
        speed_trace = []
        speed_y_mps_trace = []
        abs_acc_trace = []
        jerk_trace_mps3 = []
        ttc_trace = []
        thw_trace = []

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            ep_reward += float(reward)
            dense_ep_reward += float(info.get("eval_dense_reward", 0.0))
            ep_len += 1
            speed_trace.append(float(np.linalg.norm(eval_env.ego_state[2:4])))
            abs_acc_trace.append(float(abs(action[1]) * cfg.PHYS_ACC_MAX))
            speed_y_mps_trace.append(float(info.get("eval_vy_mps", eval_env.ego_state[3] * 0.3048)))
            jerk_trace_mps3.append(float(info.get("eval_abs_jerk_mps3", 0.0)))
            ttc_trace.append(float(info.get("eval_min_ttc", 20.0)))
            thw_trace.append(float(info.get("eval_min_thw", 10.0)))

        merge_successes.append(float(info.get("is_merge_success", False)))
        endpoint_successes.append(float(info.get("is_endpoint_success", False)))
        safety_successes.append(float(info.get("is_safety_success", False)))
        collision_flags.append(float(info.get("is_collided", False) or getattr(eval_env, "has_collided_this_episode", False)))
        episode_rewards.append(ep_reward)
        dense_episode_rewards.append(dense_ep_reward)
        episode_lengths.append(ep_len)
        mean_speeds.append(float(np.mean(speed_trace)) if speed_trace else 0.0)
        max_speeds.append(float(np.max(speed_trace)) if speed_trace else 0.0)
        mean_abs_accs.append(float(np.mean(abs_acc_trace)) if abs_acc_trace else 0.0)
        mean_abs_jerks.append(float(np.mean(np.abs(jerk_trace_mps3))) if jerk_trace_mps3 else 0.0)
        min_ttcs.append(float(np.min(ttc_trace)) if ttc_trace else 20.0)
        min_thws.append(float(np.min(thw_trace)) if thw_trace else 10.0)

        if speed_y_mps_trace and jerk_trace_mps3 and ttc_trace:
            paper_score = (
                float(np.mean(speed_y_mps_trace))
                - float(np.std(speed_y_mps_trace))
                + float(np.mean(np.clip(ttc_trace, 0.0, 20.0)))
                - 5.0 * float(np.mean(np.abs(jerk_trace_mps3)))
            )
        else:
            paper_score = -10.0
        if not info.get("is_endpoint_success", False):
            paper_score -= 5.0
        if collision_flags[-1] > 0:
            paper_score -= 5.0
        paper_scores.append(paper_score)

    dense_mean = float(np.mean(dense_episode_rewards))
    return {
        "merge_success_rate": float(np.mean(merge_successes)),
        "endpoint_success_rate": float(np.mean(endpoint_successes)),
        "safety_success_rate": float(np.mean(safety_successes)),
        "collision_rate": float(np.mean(collision_flags)),
        "eval_ep_rew_mean": float(np.mean(episode_rewards)),
        "eval_dense_return_mean": dense_mean,
        "eval_dense_return_norm100": 10.0 * dense_mean,
        "paper_rank_score_mean": float(np.mean(paper_scores)),
        "eval_ep_len_mean": float(np.mean(episode_lengths)),
        "mean_speed": float(np.mean(mean_speeds)),
        "max_speed": float(np.mean(max_speeds)),
        "mean_abs_acc": float(np.mean(mean_abs_accs)),
        "mean_abs_jerk": float(np.mean(mean_abs_jerks)),
        "min_ttc": float(np.mean(min_ttcs)),
        "min_thw": float(np.mean(min_thws)),
    }


def append_eval_metrics(log_dir, row):
    eval_csv_path = os.path.join(log_dir, "eval_metrics.csv")
    file_exists = os.path.exists(eval_csv_path)

    with open(eval_csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


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


def sanitize_run_tag(tag):
    if not tag:
        return ""
    keep = []
    for ch in tag.strip():
        if ch.isalnum() or ch in {"_", "-"}:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_")


def apply_probe_overrides(cfg):
    """Apply optional environment overrides for short probe/sweep runs."""
    overrides = {}

    def set_if_present(env_name, attr_name, caster):
        raw_value = os.environ.get(env_name)
        if raw_value is None or raw_value == "":
            return
        value = caster(raw_value)
        setattr(cfg, attr_name, value)
        overrides[attr_name] = value

    set_if_present("PROBE_EPOCHS", "EPOCHS", int)
    set_if_present("PROBE_SEED", "SEED", int)
    set_if_present("PROBE_PPO_EPOCHS", "PPO_EPOCHS", int)
    set_if_present("PROBE_PPO_MINI_BATCH_SIZE", "PPO_MINI_BATCH_SIZE", int)
    set_if_present("PROBE_GENERATOR_LR", "GENERATOR_LEARNING_RATE", float)
    set_if_present("PROBE_DISCRIMINATOR_LR", "DISCRIMINATOR_LEARNING_RATE", float)
    set_if_present("PROBE_LATE_GEN_LR_EPOCH", "PROBE_LATE_GEN_LR_EPOCH", int)
    set_if_present("PROBE_LATE_GEN_LR", "PROBE_LATE_GEN_LR", float)
    set_if_present("PROBE_LATE_N_DISC_EPOCH", "PROBE_LATE_N_DISC_EPOCH", int)
    set_if_present("PROBE_LATE_N_DISC_UPDATES", "PROBE_LATE_N_DISC_UPDATES", int)
    set_if_present("PROBE_LATE_N_DISC_RAMP_START_EPOCH", "PROBE_LATE_N_DISC_RAMP_START_EPOCH", int)
    set_if_present("PROBE_LATE_N_DISC_RAMP_END_EPOCH", "PROBE_LATE_N_DISC_RAMP_END_EPOCH", int)
    set_if_present("PROBE_BEST_SELECT_START_EPOCH", "PROBE_BEST_SELECT_START_EPOCH", int)
    set_if_present("PROBE_SAFETY_UNFREEZE_TIMESTEPS", "SAFETY_UNFREEZE_TIMESTEPS", int)
    set_if_present("PROBE_SAFETY_LIGHT_UNFREEZE_LR", "SAFETY_LIGHT_UNFREEZE_LR", float)
    set_if_present("PROBE_SAFETY_RAMP_UNFREEZE_EPOCHS", "PROBE_SAFETY_RAMP_UNFREEZE_EPOCHS", int)
    set_if_present("PROBE_SAFETY_DECAY_EPOCH", "PROBE_SAFETY_DECAY_EPOCH", int)
    set_if_present("PROBE_SAFETY_DECAY_LR", "PROBE_SAFETY_DECAY_LR", float)
    set_if_present("PROBE_SAFETY_DECAY_RAMP_EPOCHS", "PROBE_SAFETY_DECAY_RAMP_EPOCHS", int)
    set_if_present("PROBE_SAFETY_EMBED_DIM", "SAFETY_EMBED_DIM", int)
    set_if_present("PROBE_SAFETY_ORACLE_TTC_THRESHOLD", "SAFETY_ORACLE_TTC_THRESHOLD", float)
    set_if_present("PROBE_SAFETY_ORACLE_WARNING_TTC_THRESHOLD", "SAFETY_ORACLE_WARNING_TTC_THRESHOLD", float)
    set_if_present("PROBE_SAFETY_ORACLE_WARNING_WEIGHT", "SAFETY_ORACLE_WARNING_WEIGHT", float)
    set_if_present("PROBE_SAFETY_REG_COEFF", "SAFETY_REGULATOR_COEFF", float)
    set_if_present("PROBE_PREDICTIVE_SAFETY_HORIZON_STEPS", "PREDICTIVE_SAFETY_HORIZON_STEPS", int)
    set_if_present("PROBE_PREDICTIVE_SAFETY_DT", "PREDICTIVE_SAFETY_DT", float)
    set_if_present("PROBE_PREDICTIVE_SAFETY_RESIDUAL_SCALE", "PREDICTIVE_SAFETY_RESIDUAL_SCALE", float)
    set_if_present("PROBE_PREDICTIVE_SAFETY_REG_COEFF", "PREDICTIVE_SAFETY_REG_COEFF", float)
    set_if_present("PROBE_PREDICTIVE_SAFETY_GEN_PENALTY", "PREDICTIVE_SAFETY_GEN_PENALTY", float)
    set_if_present(
        "PROBE_PREDICTIVE_SAFETY_GEN_PENALTY_START_EPOCH",
        "PREDICTIVE_SAFETY_GEN_PENALTY_START_EPOCH",
        int,
    )
    set_if_present(
        "PROBE_PREDICTIVE_SAFETY_GEN_PENALTY_THRESHOLD",
        "PREDICTIVE_SAFETY_GEN_PENALTY_THRESHOLD",
        float,
    )
    set_if_present(
        "PROBE_PREDICTIVE_SAFETY_GEN_PENALTY_CLIP_MIN",
        "PREDICTIVE_SAFETY_GEN_PENALTY_CLIP_MIN",
        float,
    )
    set_if_present(
        "PROBE_PREDICTIVE_SAFETY_GEN_PENALTY_CLIP_MAX",
        "PREDICTIVE_SAFETY_GEN_PENALTY_CLIP_MAX",
        float,
    )
    set_if_present(
        "PROBE_PREDICTIVE_SAFETY_GEN_PENALTY_SOURCE",
        "PREDICTIVE_SAFETY_GEN_PENALTY_SOURCE",
        str,
    )
    set_if_present("PROBE_PREDICTIVE_SAFETY_REG_MARGIN", "PREDICTIVE_SAFETY_REG_MARGIN", float)
    set_if_present("PROBE_PREDICTIVE_SAFETY_BASE_REG_COEFF", "PREDICTIVE_SAFETY_BASE_REG_COEFF", float)
    set_if_present("PROBE_PREDICTIVE_SAFETY_LATE_REG_EPOCH", "PREDICTIVE_SAFETY_LATE_REG_EPOCH", int)
    set_if_present("PROBE_PREDICTIVE_SAFETY_LATE_REG_COEFF", "PREDICTIVE_SAFETY_LATE_REG_COEFF", float)
    set_if_present(
        "PROBE_PREDICTIVE_SAFETY_RAMP_REG_START_EPOCH",
        "PREDICTIVE_SAFETY_RAMP_REG_START_EPOCH",
        int,
    )
    set_if_present(
        "PROBE_PREDICTIVE_SAFETY_RAMP_REG_END_EPOCH",
        "PREDICTIVE_SAFETY_RAMP_REG_END_EPOCH",
        int,
    )
    set_if_present(
        "PROBE_PREDICTIVE_SAFETY_CPAIR_ADDITIVE_START_EPOCH",
        "PREDICTIVE_SAFETY_CPAIR_ADDITIVE_START_EPOCH",
        int,
    )
    set_if_present(
        "PROBE_PREDICTIVE_SAFETY_CPAIR_ADDITIVE_COEFF",
        "PREDICTIVE_SAFETY_CPAIR_ADDITIVE_COEFF",
        float,
    )
    set_if_present(
        "PROBE_PREDICTIVE_SAFETY_LATE_TINY_CPAIR_START_EPOCH",
        "PREDICTIVE_SAFETY_LATE_TINY_CPAIR_START_EPOCH",
        int,
    )
    set_if_present(
        "PROBE_PREDICTIVE_SAFETY_LATE_TINY_CPAIR_COEFF",
        "PREDICTIVE_SAFETY_LATE_TINY_CPAIR_COEFF",
        float,
    )
    set_if_present(
        "PROBE_PREDICTIVE_SAFETY_LATE_TINY_CPAIR_MARGIN",
        "PREDICTIVE_SAFETY_LATE_TINY_CPAIR_MARGIN",
        float,
    )
    predictive_residual = parse_env_bool("PROBE_ENABLE_PREDICTIVE_SAFETY_RESIDUAL")
    if predictive_residual is not None:
        cfg.ENABLE_PREDICTIVE_SAFETY_RESIDUAL = predictive_residual
        overrides["ENABLE_PREDICTIVE_SAFETY_RESIDUAL"] = predictive_residual
    predictive_cpair_additive = parse_env_bool("PROBE_PREDICTIVE_SAFETY_ENABLE_CPAIR_ADDITIVE")
    if predictive_cpair_additive is not None:
        cfg.PREDICTIVE_SAFETY_ENABLE_CPAIR_ADDITIVE = predictive_cpair_additive
        overrides["PREDICTIVE_SAFETY_ENABLE_CPAIR_ADDITIVE"] = predictive_cpair_additive
    predictive_reg_mode = os.environ.get("PROBE_PREDICTIVE_SAFETY_REG_MODE")
    if predictive_reg_mode:
        cfg.PREDICTIVE_SAFETY_REG_MODE = predictive_reg_mode
        overrides["PREDICTIVE_SAFETY_REG_MODE"] = predictive_reg_mode
    predictive_candidate_set = os.environ.get("PROBE_PREDICTIVE_SAFETY_CANDIDATE_SET")
    if predictive_candidate_set:
        cfg.PREDICTIVE_SAFETY_CANDIDATE_SET = predictive_candidate_set
        overrides["PREDICTIVE_SAFETY_CANDIDATE_SET"] = predictive_candidate_set
    predictive_safe_selection = os.environ.get("PROBE_PREDICTIVE_SAFETY_SAFE_SELECTION")
    if predictive_safe_selection:
        cfg.PREDICTIVE_SAFETY_SAFE_SELECTION = predictive_safe_selection
        overrides["PREDICTIVE_SAFETY_SAFE_SELECTION"] = predictive_safe_selection
    predictive_rank_metric = os.environ.get("PROBE_PREDICTIVE_SAFETY_RANK_METRIC")
    if predictive_rank_metric:
        cfg.PREDICTIVE_SAFETY_RANK_METRIC = predictive_rank_metric
        overrides["PREDICTIVE_SAFETY_RANK_METRIC"] = predictive_rank_metric
    predictive_focused_cpair_enable = parse_env_bool("PROBE_PREDICTIVE_SAFETY_FOCUSED_CPAIR_ENABLE")
    if predictive_focused_cpair_enable is not None:
        cfg.PREDICTIVE_SAFETY_FOCUSED_CPAIR_ENABLE = predictive_focused_cpair_enable
        overrides["PREDICTIVE_SAFETY_FOCUSED_CPAIR_ENABLE"] = predictive_focused_cpair_enable
    set_if_present(
        "PROBE_PREDICTIVE_SAFETY_FOCUSED_CPAIR_MIN_GAP",
        "PREDICTIVE_SAFETY_FOCUSED_CPAIR_MIN_GAP",
        float,
    )
    set_if_present(
        "PROBE_PREDICTIVE_SAFETY_FOCUSED_CPAIR_MIN_POLICY_RISK",
        "PREDICTIVE_SAFETY_FOCUSED_CPAIR_MIN_POLICY_RISK",
        float,
    )
    set_if_present(
        "PROBE_PREDICTIVE_SAFETY_FOCUSED_CPAIR_WEIGHT_CLIP",
        "PREDICTIVE_SAFETY_FOCUSED_CPAIR_WEIGHT_CLIP",
        float,
    )
    predictive_focused_cpair_weight_source = os.environ.get("PROBE_PREDICTIVE_SAFETY_FOCUSED_CPAIR_WEIGHT_SOURCE")
    if predictive_focused_cpair_weight_source:
        cfg.PREDICTIVE_SAFETY_FOCUSED_CPAIR_WEIGHT_SOURCE = predictive_focused_cpair_weight_source
        overrides["PREDICTIVE_SAFETY_FOCUSED_CPAIR_WEIGHT_SOURCE"] = predictive_focused_cpair_weight_source
    predictive_late_tiny_enable = parse_env_bool("PROBE_PREDICTIVE_SAFETY_LATE_TINY_CPAIR_ENABLE")
    if predictive_late_tiny_enable is not None:
        cfg.PREDICTIVE_SAFETY_LATE_TINY_CPAIR_ENABLE = predictive_late_tiny_enable
        overrides["PREDICTIVE_SAFETY_LATE_TINY_CPAIR_ENABLE"] = predictive_late_tiny_enable
    predictive_late_tiny_candidate_set = os.environ.get("PROBE_PREDICTIVE_SAFETY_LATE_TINY_CPAIR_CANDIDATE_SET")
    if predictive_late_tiny_candidate_set:
        cfg.PREDICTIVE_SAFETY_LATE_TINY_CPAIR_CANDIDATE_SET = predictive_late_tiny_candidate_set
        overrides["PREDICTIVE_SAFETY_LATE_TINY_CPAIR_CANDIDATE_SET"] = predictive_late_tiny_candidate_set
    predictive_late_tiny_safe_selection = os.environ.get("PROBE_PREDICTIVE_SAFETY_LATE_TINY_CPAIR_SAFE_SELECTION")
    if predictive_late_tiny_safe_selection:
        cfg.PREDICTIVE_SAFETY_LATE_TINY_CPAIR_SAFE_SELECTION = predictive_late_tiny_safe_selection
        overrides["PREDICTIVE_SAFETY_LATE_TINY_CPAIR_SAFE_SELECTION"] = predictive_late_tiny_safe_selection
    predictive_late_tiny_rank_metric = os.environ.get("PROBE_PREDICTIVE_SAFETY_LATE_TINY_CPAIR_RANK_METRIC")
    if predictive_late_tiny_rank_metric:
        cfg.PREDICTIVE_SAFETY_LATE_TINY_CPAIR_RANK_METRIC = predictive_late_tiny_rank_metric
        overrides["PREDICTIVE_SAFETY_LATE_TINY_CPAIR_RANK_METRIC"] = predictive_late_tiny_rank_metric
    set_if_present("PROBE_ENT_COEF", "PPO_ENT_COEF", float)
    set_if_present("PROBE_SAVE_FREQ_EPOCHS", "PROBE_SAVE_FREQ_EPOCHS", int)
    set_if_present("PROBE_QUICK_EVAL_EPISODES", "PROBE_QUICK_EVAL_EPISODES", int)
    set_if_present("PROBE_FULL_EVAL_EPISODES", "PROBE_FULL_EVAL_EPISODES", int)
    set_if_present("PROBE_FULL_EVAL_FREQ_EPOCHS", "PROBE_FULL_EVAL_FREQ_EPOCHS", int)
    set_if_present("PROBE_FULL_EVAL_PRE_END_EPOCH", "PROBE_FULL_EVAL_PRE_END_EPOCH", int)
    set_if_present("PROBE_FULL_EVAL_PRE_FREQ_EPOCHS", "PROBE_FULL_EVAL_PRE_FREQ_EPOCHS", int)
    set_if_present("PROBE_EPOCH0_EVAL_EPISODES", "PROBE_EPOCH0_EVAL_EPISODES", int)
    set_if_present("PROBE_N_DISC_UPDATES", "PROBE_N_DISC_UPDATES", int)

    reward_norm = parse_env_bool("PROBE_REWARD_NORM")
    if reward_norm is not None:
        cfg.ENABLE_REWARD_NORMALIZATION = reward_norm
        overrides["ENABLE_REWARD_NORMALIZATION"] = reward_norm

    safety_fuse_feature = parse_env_bool("PROBE_SAFETY_FUSE_FEATURE")
    if safety_fuse_feature is not None:
        cfg.SAFETY_FUSE_FEATURE = safety_fuse_feature
        overrides["SAFETY_FUSE_FEATURE"] = safety_fuse_feature

    predictive_safety = parse_env_bool("PROBE_ENABLE_PREDICTIVE_SAFETY_CRITIC")
    if predictive_safety is not None:
        cfg.ENABLE_PREDICTIVE_SAFETY_CRITIC = predictive_safety
        overrides["ENABLE_PREDICTIVE_SAFETY_CRITIC"] = predictive_safety

    predictive_candidates = parse_env_bool("PROBE_PREDICTIVE_SAFETY_USE_CANDIDATES")
    if predictive_candidates is not None:
        cfg.PREDICTIVE_SAFETY_USE_CANDIDATES = predictive_candidates
        overrides["PREDICTIVE_SAFETY_USE_CANDIDATES"] = predictive_candidates

    return overrides

def main():
    cfg = Config()
    probe_overrides = apply_probe_overrides(cfg)
    probe_tag = sanitize_run_tag(os.environ.get("PROBE_TAG", ""))
    deterministic_training = getattr(cfg, "DETERMINISTIC_TRAINING", True)
    seed_everything(cfg.SEED, deterministic=deterministic_training)
    device = torch.device("cpu")

    # ---------------------------------------------------------
    # [核心修改]: 根据消融开关挂载不同的网络架构 (严格对齐参数规模)
    # ---------------------------------------------------------
    enable_attention = getattr(cfg, 'ENABLE_ATTENTION', False)
    enable_goal = getattr(cfg, 'ENABLE_GOAL_CONDITION', False)
    enable_safety = getattr(cfg, "ENABLE_SAFETY_MODULE", False)
    enable_safety_branch = enable_safety and getattr(cfg, "ENABLE_SAFETY_BRANCH", True)
    enable_safety_aux = enable_safety and getattr(cfg, "ENABLE_SAFETY_AUX_LOSS", True)
    enable_predictive_safety = enable_safety_branch and getattr(cfg, "ENABLE_PREDICTIVE_SAFETY_CRITIC", False)
    enable_predictive_safety_residual = enable_safety_branch and getattr(cfg, "ENABLE_PREDICTIVE_SAFETY_RESIDUAL", False)
    predictive_replaces_safety = enable_predictive_safety and not enable_predictive_safety_residual
    use_predictive_safety = enable_predictive_safety or enable_predictive_safety_residual
    enable_reward_norm = getattr(cfg, "ENABLE_REWARD_NORMALIZATION", False)
    debug_use_ground_truth_reward = getattr(cfg, "DEBUG_USE_GROUND_TRUTH_REWARD", False)
    safety_fuse_feature = enable_safety and getattr(cfg, "SAFETY_FUSE_FEATURE", False)
    safety_embed_dim = int(getattr(cfg, "SAFETY_EMBED_DIM", 8 if safety_fuse_feature else 1))
    predictive_safety_horizon_steps = int(getattr(cfg, "PREDICTIVE_SAFETY_HORIZON_STEPS", 10))
    predictive_safety_dt = float(getattr(cfg, "PREDICTIVE_SAFETY_DT", cfg.DT))
    predictive_safety_use_candidates = bool(getattr(cfg, "PREDICTIVE_SAFETY_USE_CANDIDATES", True))
    predictive_safety_residual_scale = float(getattr(cfg, "PREDICTIVE_SAFETY_RESIDUAL_SCALE", 0.3))
    predictive_safety_reg_coeff = float(getattr(cfg, "PREDICTIVE_SAFETY_REG_COEFF", 0.0))
    predictive_safety_gen_penalty = float(getattr(cfg, "PREDICTIVE_SAFETY_GEN_PENALTY", 0.0))
    predictive_safety_gen_penalty_start_epoch = int(
        getattr(cfg, "PREDICTIVE_SAFETY_GEN_PENALTY_START_EPOCH", 270)
    )
    predictive_safety_gen_penalty_threshold = float(
        getattr(cfg, "PREDICTIVE_SAFETY_GEN_PENALTY_THRESHOLD", 0.7)
    )
    predictive_safety_gen_penalty_clip_min = float(
        getattr(cfg, "PREDICTIVE_SAFETY_GEN_PENALTY_CLIP_MIN", -0.03)
    )
    predictive_safety_gen_penalty_clip_max = float(
        getattr(cfg, "PREDICTIVE_SAFETY_GEN_PENALTY_CLIP_MAX", 0.0)
    )
    predictive_safety_gen_penalty_source = str(
        getattr(cfg, "PREDICTIVE_SAFETY_GEN_PENALTY_SOURCE", "q_safe_risk")
    )
    predictive_safety_reg_mode = str(getattr(cfg, "PREDICTIVE_SAFETY_REG_MODE", "candidate_pair_ranking"))
    predictive_safety_candidate_set = str(getattr(cfg, "PREDICTIVE_SAFETY_CANDIDATE_SET", "current"))
    predictive_safety_safe_selection = str(getattr(cfg, "PREDICTIVE_SAFETY_SAFE_SELECTION", "min_risk"))
    predictive_safety_rank_metric = str(getattr(cfg, "PREDICTIVE_SAFETY_RANK_METRIC", "clipped"))
    predictive_safety_reg_margin = float(getattr(cfg, "PREDICTIVE_SAFETY_REG_MARGIN", 0.2))
    predictive_safety_base_reg_coeff = getattr(cfg, "PREDICTIVE_SAFETY_BASE_REG_COEFF", None)
    if predictive_safety_base_reg_coeff is None:
        predictive_safety_base_reg_coeff = getattr(cfg, "SAFETY_REGULATOR_COEFF", 0.0)
    predictive_safety_base_reg_coeff = float(predictive_safety_base_reg_coeff)
    predictive_safety_late_reg_epoch = getattr(cfg, "PREDICTIVE_SAFETY_LATE_REG_EPOCH", None)
    predictive_safety_late_reg_coeff = getattr(cfg, "PREDICTIVE_SAFETY_LATE_REG_COEFF", None)
    if predictive_safety_late_reg_coeff is None:
        predictive_safety_late_reg_coeff = predictive_safety_base_reg_coeff
    predictive_safety_late_reg_coeff = float(predictive_safety_late_reg_coeff)
    predictive_safety_ramp_reg_start_epoch = getattr(cfg, "PREDICTIVE_SAFETY_RAMP_REG_START_EPOCH", None)
    predictive_safety_ramp_reg_end_epoch = getattr(cfg, "PREDICTIVE_SAFETY_RAMP_REG_END_EPOCH", None)
    predictive_safety_enable_cpair_additive = bool(
        getattr(cfg, "PREDICTIVE_SAFETY_ENABLE_CPAIR_ADDITIVE", False)
    )
    predictive_safety_cpair_additive_start_epoch = int(
        getattr(cfg, "PREDICTIVE_SAFETY_CPAIR_ADDITIVE_START_EPOCH", 260)
    )
    predictive_safety_cpair_additive_coeff = float(
        getattr(cfg, "PREDICTIVE_SAFETY_CPAIR_ADDITIVE_COEFF", 0.0)
    )
    predictive_safety_focused_cpair_enable = bool(
        getattr(cfg, "PREDICTIVE_SAFETY_FOCUSED_CPAIR_ENABLE", False)
    )
    predictive_safety_focused_cpair_min_gap = float(
        getattr(cfg, "PREDICTIVE_SAFETY_FOCUSED_CPAIR_MIN_GAP", 0.10)
    )
    predictive_safety_focused_cpair_min_policy_risk = float(
        getattr(cfg, "PREDICTIVE_SAFETY_FOCUSED_CPAIR_MIN_POLICY_RISK", 0.30)
    )
    predictive_safety_focused_cpair_weight_clip = float(
        getattr(cfg, "PREDICTIVE_SAFETY_FOCUSED_CPAIR_WEIGHT_CLIP", 0.20)
    )
    predictive_safety_focused_cpair_weight_source = str(
        getattr(cfg, "PREDICTIVE_SAFETY_FOCUSED_CPAIR_WEIGHT_SOURCE", "raw")
    )
    predictive_safety_late_tiny_cpair_enable = bool(
        getattr(cfg, "PREDICTIVE_SAFETY_LATE_TINY_CPAIR_ENABLE", False)
    )
    predictive_safety_late_tiny_cpair_start_epoch = int(
        getattr(cfg, "PREDICTIVE_SAFETY_LATE_TINY_CPAIR_START_EPOCH", 270)
    )
    predictive_safety_late_tiny_cpair_coeff = float(
        getattr(cfg, "PREDICTIVE_SAFETY_LATE_TINY_CPAIR_COEFF", 0.0)
    )
    predictive_safety_late_tiny_cpair_candidate_set = str(
        getattr(cfg, "PREDICTIVE_SAFETY_LATE_TINY_CPAIR_CANDIDATE_SET", "candidate_v2_leadaware")
    )
    predictive_safety_late_tiny_cpair_safe_selection = str(
        getattr(cfg, "PREDICTIVE_SAFETY_LATE_TINY_CPAIR_SAFE_SELECTION", "task_safe")
    )
    predictive_safety_late_tiny_cpair_rank_metric = str(
        getattr(cfg, "PREDICTIVE_SAFETY_LATE_TINY_CPAIR_RANK_METRIC", "clipped")
    )
    predictive_safety_late_tiny_cpair_margin = float(
        getattr(cfg, "PREDICTIVE_SAFETY_LATE_TINY_CPAIR_MARGIN", 0.1)
    )
    
    # ==========================================
    # 1. 文件夹与日志系统初始化
    # ==========================================
    os.makedirs("train_log", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    # 在日志名中加入 attention 标记，方便对比
    arch_str = "attn" if enable_attention else "mlp"
    goal_str = "goal" if enable_goal else "nogoal"
    if enable_safety:
        safety_prior_str = "safe_branch" if enable_safety_branch else "zero_branch"
        safety_aux_str = "aux" if enable_safety_aux else "noaux"
        safety_str = f"{safety_prior_str}_{safety_aux_str}"
    else:
        safety_str = "nosafe"
    stabilizer_tags = []
    if enable_reward_norm:
        stabilizer_tags.append("rnorm")
    if debug_use_ground_truth_reward:
        stabilizer_tags.append("gt")
    stabilizer_str = "_" + "_".join(stabilizer_tags) if stabilizer_tags else ""
    probe_str = f"_probe_{probe_tag}" if probe_tag else ""
    run_label = f"{arch_str}_{goal_str}_{safety_str}{stabilizer_str}{probe_str}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("train_log", f"baseline_{run_label}_{timestamp}")
    checkpoint_dir = os.path.join(log_dir, "checkpoints") if probe_tag else "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    custom_logger = init_logger(folder=log_dir, format_strs=["stdout", "csv", "log"])
    print(f"[*] 日志系统已就绪，所有训练数据将保存在: {log_dir}")
    print(f"[*] 当前网络架构: {'Attention (跳跃拼接)' if enable_attention else 'Pure MLP'}")
    print(f"[*] Repro seed={cfg.SEED}, deterministic={deterministic_training}")
    if probe_tag:
        print(f"[*] Probe tag: {probe_tag}")
    if probe_overrides:
        print(f"[*] Probe overrides: {probe_overrides}")
    print(f"[*] Reward normalization: {'NormalizedRewardNet(RunningNorm)' if enable_reward_norm else 'disabled'}")
    print(f"[*] Debug ground-truth reward: {debug_use_ground_truth_reward}")
    if enable_safety:
        print(
            "[*] 当前安全模块: 新 Q 安全模块 "
            f"(use_action={getattr(cfg, 'SAFETY_USE_ACTION', True)}, "
            f"fuse_feature={'feature_fusion' if safety_fuse_feature else 'scalar_only'}, "
            f"branch={'real' if enable_safety_branch else 'zero'}, "
            f"aux_loss={enable_safety_aux}, "
            f"reg_weight={predictive_safety_base_reg_coeff if enable_safety_aux else 0.0})"
        )
        if predictive_replaces_safety:
            print(
                "[*] Predictive safety critic enabled "
                f"(horizon_steps={predictive_safety_horizon_steps}, dt={predictive_safety_dt}, "
                f"candidates={predictive_safety_use_candidates}, reg_coeff={predictive_safety_reg_coeff}, "
                f"reg_mode={predictive_safety_reg_mode})"
            )
        elif enable_predictive_safety_residual:
            print(
                "[*] Predictive safety residual enabled "
                f"(scale={predictive_safety_residual_scale}, horizon_steps={predictive_safety_horizon_steps}, "
                f"dt={predictive_safety_dt}, candidates={predictive_safety_use_candidates}, "
                f"base_reg={predictive_safety_base_reg_coeff}, late_reg_epoch={predictive_safety_late_reg_epoch}, "
                f"late_reg_coeff={predictive_safety_late_reg_coeff}, reg_mode=legacy_aux, "
                f"gen_penalty={predictive_safety_gen_penalty}, "
                f"gen_penalty_start={predictive_safety_gen_penalty_start_epoch}, "
                f"gen_penalty_threshold={predictive_safety_gen_penalty_threshold}, "
                f"gen_penalty_clip=({predictive_safety_gen_penalty_clip_min},{predictive_safety_gen_penalty_clip_max}), "
                f"gen_penalty_source={predictive_safety_gen_penalty_source}, "
                f"cpair_additive={predictive_safety_enable_cpair_additive}, "
                f"cpair_start={predictive_safety_cpair_additive_start_epoch}, "
                f"cpair_coeff={predictive_safety_cpair_additive_coeff}, "
                f"candidate_set={predictive_safety_candidate_set}, "
                f"safe_selection={predictive_safety_safe_selection}, "
                f"rank_metric={predictive_safety_rank_metric}, "
                f"focused_cpair={predictive_safety_focused_cpair_enable}, "
                f"focused_gap={predictive_safety_focused_cpair_min_gap}, "
                f"focused_policy_risk={predictive_safety_focused_cpair_min_policy_risk}, "
                f"focused_weight_clip={predictive_safety_focused_cpair_weight_clip}, "
                f"focused_weight_source={predictive_safety_focused_cpair_weight_source})"
            )
    else:
        print("[*] 当前安全模块: 关闭")

    # ==========================================
    # 2. 加载与处理数据集
    # ==========================================
    data_paths = [
        "data/lane_change_trajectories-0750am-0805am",
        "data/lane_change_trajectories-0805am-0820am",
        "data/lane_change_trajectories-0820am-0835am"
    ]
    
    print("正在加载数据集，请稍候...")
    dataset = MergingDataset(data_paths, device=device)
    
    if len(dataset) == 0:
        print("Error: No trajectories found across all specified data paths.")
        sys.exit(1)
        
    print(f"成功加载 {len(dataset)} 条专家轨迹！")
    train_dataset, val_dataset = split_dataset(dataset, train_ratio=0.8, seed=cfg.SEED)
    print(f"[*] 数据集划分: train={len(train_dataset)} | val={len(val_dataset)}")
    expert_trajectories = convert_to_trajectories(train_dataset, cfg)

    safety_net = None
    predictive_safety_net = None
    predictive_safety_oracle = None
    if enable_safety:
        if enable_safety_branch:
            print("[*] 正在预训练安全网络...")
            if predictive_replaces_safety:
                safety_oracle = PredictiveSafetyOracle(
                    cfg,
                    train_dataset.expert_mean,
                    train_dataset.expert_std,
                    horizon_steps=predictive_safety_horizon_steps,
                    dt=predictive_safety_dt,
                )
                safety_net = PredictiveSafetyCostNetwork(
                    state_dim=16,
                    action_dim=2,
                    hidden_dim=128,
                    use_action=getattr(cfg, "SAFETY_USE_ACTION", True),
                )
                pretrain_stats = pretrain_predictive_safety_network(
                    safety_net,
                    train_dataset,
                    safety_oracle,
                    device=device,
                    epochs=15,
                    batch_size=512,
                    lr=cfg.SAFETY_LEARNING_RATE,
                    use_candidates=predictive_safety_use_candidates,
                    seed=cfg.SEED,
                    verbose=True,
                )
            else:
                safety_oracle = SafetyOracleQ(cfg, train_dataset.expert_mean, train_dataset.expert_std)
                safety_net = SafetyQNetwork(
                    state_dim=16,
                    action_dim=2,
                    hidden_dim=128,
                    use_action=getattr(cfg, "SAFETY_USE_ACTION", True),
                )
                pretrain_stats = pretrain_safety_q_network(
                    safety_net,
                    train_dataset,
                    safety_oracle,
                    device=device,
                    epochs=15,
                    batch_size=512,
                    lr=cfg.SAFETY_LEARNING_RATE,
                    synthetic_multiplier=1.0,
                    seed=cfg.SEED,
                    verbose=True,
                )
                if enable_predictive_safety_residual:
                    print("[*] 正在预训练 predictive residual 安全网络...")
                    predictive_safety_oracle = PredictiveSafetyOracle(
                        cfg,
                        train_dataset.expert_mean,
                        train_dataset.expert_std,
                        horizon_steps=predictive_safety_horizon_steps,
                        dt=predictive_safety_dt,
                    )
                    predictive_safety_net = PredictiveSafetyCostNetwork(
                        state_dim=16,
                        action_dim=2,
                        hidden_dim=128,
                        use_action=getattr(cfg, "SAFETY_USE_ACTION", True),
                    )
                    predictive_pretrain_stats = pretrain_predictive_safety_network(
                        predictive_safety_net,
                        train_dataset,
                        predictive_safety_oracle,
                        device=device,
                        epochs=15,
                        batch_size=512,
                        lr=cfg.SAFETY_LEARNING_RATE,
                        use_candidates=predictive_safety_use_candidates,
                        seed=cfg.SEED,
                        verbose=True,
                    )
                    print(f"[*] Predictive residual pretrain done: {predictive_pretrain_stats}")
            print(f"[*] Safety pretrain done: {pretrain_stats}")
        else:
            print("[*] 安全先验分支消融: 使用恒零安全网络，跳过安全预训练。")
            safety_net = ZeroSafetyQNetwork(
                state_dim=16,
                action_dim=2,
                feature_dim=128,
                use_action=getattr(cfg, "SAFETY_USE_ACTION", True),
            )

    # ==========================================
    # 3. 初始化环境与网络
    # ==========================================
    def make_train_env():
        raw_env = MergingEnv(train_dataset)
        return Monitor(
            raw_env,
            info_keywords=("is_success", "is_merge_success", "is_endpoint_success", "is_safety_success"),
        )

    env = DummyVecEnv([make_train_env])
    env.seed(cfg.SEED)
    divider_x = cfg.X_MIN + cfg.LANE_WIDTH
    goal_bonus = 0.5

    # ---------------------------------------------------------
    # [核心修改]: 根据消融开关挂载不同的网络架构
    # ---------------------------------------------------------
    
    if enable_attention:
        # ====================================================
        # A. 开启注意力机制 (Attention Skip-Connection)
        # ====================================================
        print("[*] 正在加载: 多头注意力机制架构 (Attention ON)")
        
        # 1. 判别器：使用 AttentionRewardNet，其内部已对齐 128x128
        if enable_safety:
            base_reward_net = SafeQAttentionRewardNet(
                observation_space=env.observation_space,
                action_space=env.action_space,
                safety_net=safety_net,
                hidden_dim=64,
                safety_embed_dim=safety_embed_dim,
                freeze_safety=True,
                fuse_safety_feature=safety_fuse_feature,
                predictive_safety_net=predictive_safety_net,
                predictive_residual_scale=predictive_safety_residual_scale if enable_predictive_safety_residual else 0.0,
                freeze_predictive_safety=True,
            )
        else:
            base_reward_net = AttentionRewardNet(
                observation_space=env.observation_space,
                action_space=env.action_space,
                hidden_dim=64  # 注意力输出维度
            )
        reward_net = GoalRewardWrapper(
            base_reward_net,
            expert_mean_x=train_dataset.expert_mean[0],
            expert_std_x=train_dataset.expert_std[0],
            divider_x=divider_x,
            goal_bonus=goal_bonus,
            high_risk_penalty_lambda=0.0,
            high_risk_threshold=predictive_safety_gen_penalty_threshold,
            high_risk_penalty_clip_min=predictive_safety_gen_penalty_clip_min,
            high_risk_penalty_clip_max=predictive_safety_gen_penalty_clip_max,
            high_risk_source=predictive_safety_gen_penalty_source,
            predictive_safety_oracle=predictive_safety_oracle,
        )
        goal_reward_wrapper = reward_net
        
        # 2. 生成器 (PPO)：提取特征后，送入严格对齐的 128x128 网络
        policy_kwargs = dict(
            features_extractor_class=AttentionFeaturesExtractor,
            features_extractor_kwargs=dict(hidden_dim=64),
            net_arch=dict(pi=[128, 128], vf=[128, 128]), # <--- 绝对对齐
            activation_fn=nn.Tanh
        )
    else:
        # ====================================================
        # B. 关闭注意力机制 (Pure MLP Baseline)
        # ====================================================
        print("[*] 正在加载: 纯多层感知机架构 (Attention OFF)")
        
        # 1. 判别器：使用原生 BasicRewardNet，并指定 128x128
        if enable_safety:
            base_reward_net = SafeQMLPRewardNet(
                observation_space=env.observation_space,
                action_space=env.action_space,
                safety_net=safety_net,
                safety_embed_dim=safety_embed_dim,
                freeze_safety=True,
                fuse_safety_feature=safety_fuse_feature,
                predictive_safety_net=predictive_safety_net,
                predictive_residual_scale=predictive_safety_residual_scale if enable_predictive_safety_residual else 0.0,
                freeze_predictive_safety=True,
            )
        else:
            base_reward_net = BasicRewardNet(
                observation_space=env.observation_space,
                action_space=env.action_space,
                normalize_input_layer=nn.Identity,
                hid_sizes=(128, 128),  # <--- 绝对对齐
                use_state=True,        
                use_action=True,       
                use_next_state=False,
                use_done=False,
            )
        reward_net = GoalRewardWrapper(
            base_reward_net,
            expert_mean_x=train_dataset.expert_mean[0],
            expert_std_x=train_dataset.expert_std[0],
            divider_x=divider_x,
            goal_bonus=goal_bonus,
            high_risk_penalty_lambda=0.0,
            high_risk_threshold=predictive_safety_gen_penalty_threshold,
            high_risk_penalty_clip_min=predictive_safety_gen_penalty_clip_min,
            high_risk_penalty_clip_max=predictive_safety_gen_penalty_clip_max,
            high_risk_source=predictive_safety_gen_penalty_source,
            predictive_safety_oracle=predictive_safety_oracle,
        )
        goal_reward_wrapper = reward_net
        
        # 2. 生成器 (PPO)：直接送入严格对齐的 128x128 网络
        policy_kwargs = dict(
            activation_fn=nn.Tanh,  
            net_arch=dict(pi=[128, 128], vf=[128, 128])  # <--- 绝对对齐
        )
        if enable_goal:
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
    # ---------------------------------------------------------

    if enable_reward_norm:
        reward_net = NormalizedRewardNet(reward_net, normalize_output_layer=RunningNorm)

    # 实例化 PPO 学习器
    learner = PPO(
        env=env,
        policy=MlpPolicy, # 这里的 MlpPolicy 会根据上面的 policy_kwargs 自动挂载
        batch_size=cfg.PPO_MINI_BATCH_SIZE,
        learning_rate=cfg.GENERATOR_LEARNING_RATE,
        n_epochs=cfg.PPO_EPOCHS,
        ent_coef=getattr(cfg, "PPO_ENT_COEF", 0.005),
        clip_range=0.2,
        target_kl=0.01,         
        policy_kwargs=policy_kwargs,
        seed=cfg.SEED,
    )

    n_disc_updates_per_round = getattr(cfg, "PROBE_N_DISC_UPDATES", 5)
    base_n_disc_updates = n_disc_updates_per_round
    trainer_kwargs = dict(
        demonstrations=expert_trajectories,
        demo_batch_size=256,
        gen_replay_buffer_capacity=2048,
        n_disc_updates_per_round=n_disc_updates_per_round,
        venv=env,
        gen_algo=learner,
        reward_net=reward_net,
        debug_use_ground_truth=debug_use_ground_truth_reward,
        allow_variable_horizon=True,
        custom_logger=custom_logger,
        disc_opt_kwargs=dict(
            lr=cfg.DISCRIMINATOR_LEARNING_RATE,
            weight_decay=1e-3,
        ),
    )
    if enable_safety:
        safety_loss_weight = 0.0
        safety_reg_mode = "legacy_aux"
        safety_cpair_additive_weight = 0.0
        safety_late_tiny_cpair_weight = 0.0
        if enable_safety_aux:
            if enable_predictive_safety and not enable_predictive_safety_residual:
                safety_loss_weight = predictive_safety_reg_coeff
                safety_reg_mode = predictive_safety_reg_mode
            else:
                safety_loss_weight = predictive_safety_base_reg_coeff
                safety_reg_mode = "legacy_aux"
                if enable_predictive_safety_residual and predictive_safety_enable_cpair_additive:
                    safety_cpair_additive_weight = 0.0
        airl_trainer = MildSafetyAIRL(
            **trainer_kwargs,
            safety_loss_weight=safety_loss_weight,
            safety_reg_mode=safety_reg_mode,
            safety_reg_margin=predictive_safety_reg_margin,
            safety_cpair_additive_weight=safety_cpair_additive_weight,
            safety_candidate_set=predictive_safety_candidate_set,
            safety_safe_selection=predictive_safety_safe_selection,
            safety_rank_metric=predictive_safety_rank_metric,
            safety_focused_cpair_enable=predictive_safety_focused_cpair_enable,
            safety_focused_cpair_min_gap=predictive_safety_focused_cpair_min_gap,
            safety_focused_cpair_min_policy_risk=predictive_safety_focused_cpair_min_policy_risk,
            safety_focused_cpair_weight_clip=predictive_safety_focused_cpair_weight_clip,
            safety_focused_cpair_weight_source=predictive_safety_focused_cpair_weight_source,
            safety_late_tiny_cpair_weight=safety_late_tiny_cpair_weight,
            safety_late_tiny_candidate_set=predictive_safety_late_tiny_cpair_candidate_set,
            safety_late_tiny_safe_selection=predictive_safety_late_tiny_cpair_safe_selection,
            safety_late_tiny_rank_metric=predictive_safety_late_tiny_cpair_rank_metric,
            safety_late_tiny_reg_margin=predictive_safety_late_tiny_cpair_margin,
        )
    else:
        airl_trainer = AIRL(**trainer_kwargs)

    # ==========================================
    # 4. 切块训练与定期保存 (Chunked Training)
    # ==========================================
    save_freq_epochs = getattr(cfg, "PROBE_SAVE_FREQ_EPOCHS", 10)
    total_epochs = cfg.EPOCHS
    if total_epochs % save_freq_epochs != 0:
        raise ValueError(
            f"EPOCHS ({total_epochs}) must be divisible by save_freq_epochs ({save_freq_epochs}) "
            "for the current chunked trainer."
        )
    chunks = total_epochs // save_freq_epochs
    steps_per_chunk = save_freq_epochs * cfg.STEPS_PER_EPOCH
    base_gen_lr = cfg.GENERATOR_LEARNING_RATE
    late_gen_lr_epoch = getattr(cfg, "PROBE_LATE_GEN_LR_EPOCH", None)
    late_gen_lr = getattr(cfg, "PROBE_LATE_GEN_LR", None)
    late_n_disc_epoch = getattr(cfg, "PROBE_LATE_N_DISC_EPOCH", None)
    late_n_disc_updates = getattr(cfg, "PROBE_LATE_N_DISC_UPDATES", None)
    late_n_disc_ramp_start_epoch = getattr(cfg, "PROBE_LATE_N_DISC_RAMP_START_EPOCH", None)
    late_n_disc_ramp_end_epoch = getattr(cfg, "PROBE_LATE_N_DISC_RAMP_END_EPOCH", None)
    best_select_start_epoch = getattr(cfg, "PROBE_BEST_SELECT_START_EPOCH", 0)
    eval_freq_epochs = 1
    quick_eval_episodes = getattr(cfg, "PROBE_QUICK_EVAL_EPISODES", 8)
    full_eval_episodes = getattr(cfg, "PROBE_FULL_EVAL_EPISODES", 40)
    full_eval_freq_epochs = getattr(cfg, "PROBE_FULL_EVAL_FREQ_EPOCHS", save_freq_epochs)
    full_eval_pre_end_epoch = getattr(cfg, "PROBE_FULL_EVAL_PRE_END_EPOCH", 0)
    full_eval_pre_freq_epochs = getattr(cfg, "PROBE_FULL_EVAL_PRE_FREQ_EPOCHS", full_eval_freq_epochs)
    epoch0_eval_episodes = getattr(cfg, "PROBE_EPOCH0_EVAL_EPISODES", full_eval_episodes)
    if full_eval_freq_epochs <= 0:
        raise ValueError("PROBE_FULL_EVAL_FREQ_EPOCHS must be >= 1")
    if full_eval_pre_freq_epochs <= 0:
        raise ValueError("PROBE_FULL_EVAL_PRE_FREQ_EPOCHS must be >= 1")
    if epoch0_eval_episodes <= 0:
        raise ValueError("PROBE_EPOCH0_EVAL_EPISODES must be >= 1")
    n_eval_episodes = full_eval_episodes
    safety_unfreeze_timesteps = getattr(cfg, "SAFETY_UNFREEZE_TIMESTEPS", 100000)
    safety_light_unfreeze_lr = getattr(cfg, "SAFETY_LIGHT_UNFREEZE_LR", 1e-5)
    safety_ramp_unfreeze_epochs = getattr(cfg, "PROBE_SAFETY_RAMP_UNFREEZE_EPOCHS", 0)
    safety_decay_epoch = getattr(cfg, "PROBE_SAFETY_DECAY_EPOCH", None)
    safety_decay_lr = getattr(cfg, "PROBE_SAFETY_DECAY_LR", None)
    safety_decay_ramp_epochs = getattr(cfg, "PROBE_SAFETY_DECAY_RAMP_EPOCHS", 0)
    safety_phase = "frozen" if enable_safety else "disabled"
    safety_current_grad_scale = 0.0
    safety_aux_current_weight = float(safety_loss_weight if enable_safety else 0.0)
    safety_cpair_current_weight = float(safety_cpair_additive_weight if enable_safety else 0.0)
    safety_late_tiny_cpair_current_weight = float(safety_late_tiny_cpair_weight if enable_safety else 0.0)
    gen_risk_penalty_current_lambda = 0.0
    total_train_timesteps = total_epochs * cfg.STEPS_PER_EPOCH
    safety_grad_scale = min(
        1.0,
        safety_light_unfreeze_lr / max(float(cfg.DISCRIMINATOR_LEARNING_RATE), 1e-12),
    )
    safety_decay_grad_scale = None
    if safety_decay_epoch is not None and safety_decay_lr is not None:
        safety_decay_grad_scale = min(
            1.0,
            safety_decay_lr / max(float(cfg.DISCRIMINATOR_LEARNING_RATE), 1e-12),
        )
    save_run_metadata(
        log_dir,
        {
            "reference_run": "baseline_attn_20260412_183104",
            "config": config_to_dict(cfg),
            "effective_params": {
                "probe_tag": probe_tag,
                "probe_overrides": probe_overrides,
                "checkpoint_dir": checkpoint_dir,
                "seed": cfg.SEED,
                "deterministic_training": deterministic_training,
                "n_disc_updates_per_round": n_disc_updates_per_round,
                "ppo_mini_batch_size": cfg.PPO_MINI_BATCH_SIZE,
                "ppo_epochs": cfg.PPO_EPOCHS,
                "ppo_ent_coef": getattr(cfg, "PPO_ENT_COEF", 0.005),
                "generator_learning_rate": cfg.GENERATOR_LEARNING_RATE,
                "discriminator_learning_rate": cfg.DISCRIMINATOR_LEARNING_RATE,
                "late_gen_lr_epoch": late_gen_lr_epoch,
                "late_gen_lr": late_gen_lr,
                "late_n_disc_epoch": late_n_disc_epoch,
                "late_n_disc_updates": late_n_disc_updates,
                "late_n_disc_ramp_start_epoch": late_n_disc_ramp_start_epoch,
                "late_n_disc_ramp_end_epoch": late_n_disc_ramp_end_epoch,
                "best_select_start_epoch": best_select_start_epoch,
                "save_freq_epochs": save_freq_epochs,
                "eval_freq_epochs": eval_freq_epochs,
                "quick_eval_episodes": quick_eval_episodes,
                "full_eval_episodes": full_eval_episodes,
                "full_eval_freq_epochs": full_eval_freq_epochs,
                "full_eval_pre_end_epoch": full_eval_pre_end_epoch,
                "full_eval_pre_freq_epochs": full_eval_pre_freq_epochs,
                "epoch0_eval_episodes": epoch0_eval_episodes,
                "n_eval_episodes": n_eval_episodes,
                "goal_bonus": goal_bonus,
                "attention_query_uses_goal": False,
                "safety_enabled": enable_safety,
                "safety_branch_enabled": enable_safety_branch,
                "safety_aux_enabled": enable_safety_aux,
                "predictive_safety_enabled": use_predictive_safety,
                "predictive_safety_replaces_safety": predictive_replaces_safety,
                "predictive_safety_residual_enabled": enable_predictive_safety_residual,
                "predictive_safety_residual_scale": predictive_safety_residual_scale if enable_predictive_safety_residual else 0.0,
                "predictive_safety_reg_mode": predictive_safety_reg_mode,
                "predictive_safety_candidate_set": predictive_safety_candidate_set,
                "predictive_safety_safe_selection": predictive_safety_safe_selection,
                "predictive_safety_rank_metric": predictive_safety_rank_metric,
                  "predictive_safety_base_reg_coeff": predictive_safety_base_reg_coeff,
                  "predictive_safety_gen_penalty": predictive_safety_gen_penalty,
                  "predictive_safety_gen_penalty_start_epoch": predictive_safety_gen_penalty_start_epoch,
                  "predictive_safety_gen_penalty_threshold": predictive_safety_gen_penalty_threshold,
                  "predictive_safety_gen_penalty_clip_min": predictive_safety_gen_penalty_clip_min,
                  "predictive_safety_gen_penalty_clip_max": predictive_safety_gen_penalty_clip_max,
                  "predictive_safety_gen_penalty_source": predictive_safety_gen_penalty_source,
                  "predictive_safety_late_reg_epoch": predictive_safety_late_reg_epoch,
                "predictive_safety_late_reg_coeff": predictive_safety_late_reg_coeff,
                "predictive_safety_ramp_reg_start_epoch": predictive_safety_ramp_reg_start_epoch,
                "predictive_safety_ramp_reg_end_epoch": predictive_safety_ramp_reg_end_epoch,
                "predictive_safety_enable_cpair_additive": predictive_safety_enable_cpair_additive,
                "predictive_safety_cpair_additive_start_epoch": predictive_safety_cpair_additive_start_epoch,
                "predictive_safety_cpair_additive_coeff": predictive_safety_cpair_additive_coeff,
                "predictive_safety_focused_cpair_enable": predictive_safety_focused_cpair_enable,
                "predictive_safety_focused_cpair_min_gap": predictive_safety_focused_cpair_min_gap,
                "predictive_safety_focused_cpair_min_policy_risk": predictive_safety_focused_cpair_min_policy_risk,
                "predictive_safety_focused_cpair_weight_clip": predictive_safety_focused_cpair_weight_clip,
                "predictive_safety_focused_cpair_weight_source": predictive_safety_focused_cpair_weight_source,
                "predictive_safety_late_tiny_cpair_enable": predictive_safety_late_tiny_cpair_enable,
                "predictive_safety_late_tiny_cpair_start_epoch": predictive_safety_late_tiny_cpair_start_epoch,
                "predictive_safety_late_tiny_cpair_coeff": predictive_safety_late_tiny_cpair_coeff,
                "predictive_safety_late_tiny_cpair_candidate_set": predictive_safety_late_tiny_cpair_candidate_set,
                "predictive_safety_late_tiny_cpair_safe_selection": predictive_safety_late_tiny_cpair_safe_selection,
                "predictive_safety_late_tiny_cpair_rank_metric": predictive_safety_late_tiny_cpair_rank_metric,
                "predictive_safety_late_tiny_cpair_margin": predictive_safety_late_tiny_cpair_margin,
                "safety_fusion_effective": "feature_fusion" if safety_fuse_feature else ("scalar_only" if enable_safety else "disabled"),
                "safety_embed_dim": safety_embed_dim if enable_safety else 0,
                "safety_loss_weight_effective": (
                    predictive_safety_reg_coeff
                    if (enable_safety_aux and enable_predictive_safety)
                    else (predictive_safety_base_reg_coeff if enable_safety_aux else 0.0)
                ),
                "reward_normalization_enabled": enable_reward_norm,
                "reward_normalization_layer": "RunningNorm" if enable_reward_norm else "disabled",
                "debug_use_ground_truth_reward": debug_use_ground_truth_reward,
                "safety_phase_initial": safety_phase,
                "safety_unfreeze_timesteps": safety_unfreeze_timesteps,
                "safety_light_unfreeze_lr": safety_light_unfreeze_lr,
                "safety_target_grad_scale": safety_grad_scale,
                "safety_ramp_unfreeze_epochs": safety_ramp_unfreeze_epochs,
                "safety_decay_epoch": safety_decay_epoch,
                "safety_decay_lr": safety_decay_lr,
                "safety_decay_ramp_epochs": safety_decay_ramp_epochs,
                "safety_decay_grad_scale": safety_decay_grad_scale,
            },
        },
    )

    if enable_safety and hasattr(base_reward_net, "set_safety_training_phase") and not predictive_replaces_safety:
        base_reward_net.set_safety_training_phase("frozen")
        if safety_unfreeze_timesteps > total_train_timesteps:
            print(
                "[*] Safety schedule: frozen_only "
                f"(unfreeze_after={safety_unfreeze_timesteps} > total_train_timesteps={total_train_timesteps})"
            )
        else:
            if safety_ramp_unfreeze_epochs > 0:
                print(
                    "[*] Safety schedule: freeze -> ramp_unfreeze -> light_unfreeze "
                    f"(unfreeze_after={safety_unfreeze_timesteps}, "
                    f"ramp_epochs={safety_ramp_unfreeze_epochs}, "
                    f"safety_lr~{safety_light_unfreeze_lr:.1e})"
                )
            else:
                print(
                    "[*] Safety schedule: freeze -> light_unfreeze "
                    f"(unfreeze_after={safety_unfreeze_timesteps}, "
                    f"safety_lr~{safety_light_unfreeze_lr:.1e})"
                )
            if safety_decay_epoch is not None and safety_decay_grad_scale is not None:
                print(
                    "[*] Safety late decay enabled "
                    f"(decay_epoch={safety_decay_epoch}, "
                    f"decay_lr~{safety_decay_lr:.1e}, "
                    f"decay_grad_scale={safety_decay_grad_scale:.3f}, "
                    f"decay_ramp_epochs={safety_decay_ramp_epochs})"
                )

    if enable_safety and enable_safety_aux and isinstance(airl_trainer, MildSafetyAIRL) and not predictive_replaces_safety:
        airl_trainer.set_safety_aux_weights(
            legacy_weight=safety_aux_current_weight,
            candidate_pair_additive_weight=safety_cpair_current_weight,
            late_tiny_candidate_pair_weight=safety_late_tiny_cpair_current_weight,
        )
        print(
            "[*] Safety aux schedule: "
            f"legacy_base={predictive_safety_base_reg_coeff:.3f}, "
            f"legacy_late_epoch={predictive_safety_late_reg_epoch}, "
            f"legacy_late_coeff={predictive_safety_late_reg_coeff:.3f}, "
            f"legacy_ramp=({predictive_safety_ramp_reg_start_epoch}->{predictive_safety_ramp_reg_end_epoch}), "
            f"cpair_additive={predictive_safety_enable_cpair_additive}, "
            f"cpair_start={predictive_safety_cpair_additive_start_epoch}, "
            f"cpair_coeff={predictive_safety_cpair_additive_coeff:.3f}, "
            f"late_tiny_cpair={predictive_safety_late_tiny_cpair_enable}, "
            f"late_tiny_start={predictive_safety_late_tiny_cpair_start_epoch}, "
            f"late_tiny_coeff={predictive_safety_late_tiny_cpair_coeff:.3f}, "
            f"late_tiny_set={predictive_safety_late_tiny_cpair_candidate_set}, "
              f"late_tiny_sel={predictive_safety_late_tiny_cpair_safe_selection}, "
              f"late_tiny_rank={predictive_safety_late_tiny_cpair_rank_metric}"
          )

    def update_generator_penalty_for_epoch(epoch_to_train):
        nonlocal gen_risk_penalty_current_lambda
        if not hasattr(goal_reward_wrapper, "set_high_risk_penalty"):
            return
        next_lambda = 0.0
        if (
            enable_predictive_safety_residual
            and predictive_safety_gen_penalty > 0.0
            and epoch_to_train >= predictive_safety_gen_penalty_start_epoch
        ):
            next_lambda = predictive_safety_gen_penalty
        if abs(next_lambda - gen_risk_penalty_current_lambda) < 1e-8:
            return
        goal_reward_wrapper.set_high_risk_penalty(
            lambda_risk=next_lambda,
            threshold=predictive_safety_gen_penalty_threshold,
            clip_min=predictive_safety_gen_penalty_clip_min,
            clip_max=predictive_safety_gen_penalty_clip_max,
            source=predictive_safety_gen_penalty_source,
        )
        gen_risk_penalty_current_lambda = next_lambda
        print(
            "[*] Generator risk penalty updated "
            f"(epoch_to_train={epoch_to_train}, lambda={gen_risk_penalty_current_lambda:.4f}, "
            f"threshold={predictive_safety_gen_penalty_threshold:.2f}, "
            f"source={predictive_safety_gen_penalty_source}, "
            f"clip=({predictive_safety_gen_penalty_clip_min:.3f},{predictive_safety_gen_penalty_clip_max:.3f}))"
        )

    def reset_generator_penalty_stats():
        if hasattr(goal_reward_wrapper, "reset_high_risk_penalty_stats"):
            goal_reward_wrapper.reset_high_risk_penalty_stats()

    def get_generator_penalty_stats(reset=False):
        if hasattr(goal_reward_wrapper, "get_high_risk_penalty_stats"):
            return goal_reward_wrapper.get_high_risk_penalty_stats(reset=reset)
        return {
            "gen_risk_penalty_calls": 0,
            "gen_risk_penalty_samples": 0,
            "gen_q_safe_risk_mean": 0.0,
            "gen_q_safe_risk_max": 0.0,
            "gen_q_safe_risk_over_threshold_rate": 0.0,
            "gen_risk_penalty_mean": 0.0,
            "gen_risk_penalty_abs_mean": 0.0,
            "gen_risk_penalty_min": 0.0,
            "gen_risk_penalty_nonzero_rate": 0.0,
            "gen_risk_penalty_clip_min_rate": 0.0,
            "gen_risk_penalty_active_lambda": gen_risk_penalty_current_lambda,
            "gen_risk_penalty_active_threshold": predictive_safety_gen_penalty_threshold,
            "gen_risk_penalty_source": predictive_safety_gen_penalty_source,
        }

    def update_safety_phase_for_epoch(epoch_to_train):
        nonlocal safety_phase, safety_current_grad_scale
        if not (
            enable_safety
            and hasattr(base_reward_net, "set_safety_training_phase")
            and not predictive_replaces_safety
            and safety_unfreeze_timesteps <= total_train_timesteps
        ):
            return

        unfreeze_epoch = safety_unfreeze_timesteps / float(cfg.STEPS_PER_EPOCH)
        if epoch_to_train <= unfreeze_epoch:
            return

        if safety_ramp_unfreeze_epochs > 0:
            ramp_fraction = (epoch_to_train - unfreeze_epoch) / float(safety_ramp_unfreeze_epochs)
            ramp_fraction = min(1.0, max(0.0, ramp_fraction))
            next_grad_scale = safety_grad_scale * ramp_fraction
            next_phase = "ramp_unfreeze" if ramp_fraction < 1.0 else "light_unfreeze"
        else:
            next_grad_scale = safety_grad_scale
            next_phase = "light_unfreeze"

        if (
            safety_decay_epoch is not None
            and safety_decay_grad_scale is not None
            and epoch_to_train >= safety_decay_epoch
        ):
            if safety_decay_ramp_epochs and safety_decay_ramp_epochs > 0:
                decay_fraction = (epoch_to_train - safety_decay_epoch) / float(safety_decay_ramp_epochs)
                decay_fraction = min(1.0, max(0.0, decay_fraction))
                next_grad_scale = (
                    safety_grad_scale
                    + decay_fraction * (safety_decay_grad_scale - safety_grad_scale)
                )
                next_phase = "decay_ramp_unfreeze" if decay_fraction < 1.0 else "decay_unfreeze"
            else:
                next_grad_scale = safety_decay_grad_scale
                next_phase = "decay_unfreeze"

        if next_grad_scale <= 0.0:
            return
        if safety_phase == next_phase and abs(safety_current_grad_scale - next_grad_scale) < 1e-8:
            return

        base_reward_net.set_safety_training_phase("light_unfreeze", grad_scale=next_grad_scale)
        safety_phase = next_phase
        safety_current_grad_scale = next_grad_scale
        print(
            "[*] Safety schedule updated "
            f"(epoch_to_train={epoch_to_train}, phase={safety_phase}, "
            f"grad_scale={safety_current_grad_scale:.3f})"
        )

    def update_safety_aux_schedule_for_epoch(epoch_to_train):
        nonlocal safety_aux_current_weight, safety_cpair_current_weight, safety_late_tiny_cpair_current_weight
        if not (
            enable_safety
            and enable_safety_aux
            and isinstance(airl_trainer, MildSafetyAIRL)
            and not predictive_replaces_safety
        ):
            return

        next_legacy_weight = predictive_safety_base_reg_coeff
        if (
            predictive_safety_ramp_reg_start_epoch is not None
            and predictive_safety_ramp_reg_end_epoch is not None
            and predictive_safety_ramp_reg_end_epoch > predictive_safety_ramp_reg_start_epoch
        ):
            if epoch_to_train >= predictive_safety_ramp_reg_end_epoch:
                next_legacy_weight = predictive_safety_late_reg_coeff
            elif epoch_to_train >= predictive_safety_ramp_reg_start_epoch:
                ramp_fraction = (
                    (epoch_to_train - predictive_safety_ramp_reg_start_epoch)
                    / float(predictive_safety_ramp_reg_end_epoch - predictive_safety_ramp_reg_start_epoch)
                )
                ramp_fraction = min(1.0, max(0.0, ramp_fraction))
                next_legacy_weight = (
                    predictive_safety_base_reg_coeff
                    + ramp_fraction * (predictive_safety_late_reg_coeff - predictive_safety_base_reg_coeff)
                )
        elif (
            predictive_safety_late_reg_epoch is not None
            and epoch_to_train >= predictive_safety_late_reg_epoch
        ):
            next_legacy_weight = predictive_safety_late_reg_coeff

        next_cpair_weight = 0.0
        if (
            enable_predictive_safety_residual
            and predictive_safety_enable_cpair_additive
            and epoch_to_train >= predictive_safety_cpair_additive_start_epoch
        ):
            next_cpair_weight = predictive_safety_cpair_additive_coeff

        next_late_tiny_cpair_weight = 0.0
        if (
            enable_predictive_safety_residual
            and predictive_safety_late_tiny_cpair_enable
            and epoch_to_train >= predictive_safety_late_tiny_cpair_start_epoch
        ):
            next_late_tiny_cpair_weight = predictive_safety_late_tiny_cpair_coeff

        if (
            abs(next_legacy_weight - safety_aux_current_weight) < 1e-8
            and abs(next_cpair_weight - safety_cpair_current_weight) < 1e-8
            and abs(next_late_tiny_cpair_weight - safety_late_tiny_cpair_current_weight) < 1e-8
        ):
            return

        airl_trainer.set_safety_aux_weights(
            legacy_weight=next_legacy_weight,
            candidate_pair_additive_weight=next_cpair_weight,
            late_tiny_candidate_pair_weight=next_late_tiny_cpair_weight,
        )
        cfg.SAFETY_REGULATOR_COEFF = next_legacy_weight
        safety_aux_current_weight = next_legacy_weight
        safety_cpair_current_weight = next_cpair_weight
        safety_late_tiny_cpair_current_weight = next_late_tiny_cpair_weight
        print(
            "[*] Safety aux schedule updated "
            f"(epoch_to_train={epoch_to_train}, "
            f"legacy_weight={safety_aux_current_weight:.3f}, "
            f"cpair_additive_weight={safety_cpair_current_weight:.3f}, "
            f"late_tiny_cpair_weight={safety_late_tiny_cpair_current_weight:.3f})"
        )

    def get_n_disc_updates_for_epoch(epoch_to_train):
        if late_n_disc_updates is None:
            return base_n_disc_updates

        if (
            late_n_disc_ramp_start_epoch is not None
            and late_n_disc_ramp_end_epoch is not None
            and late_n_disc_ramp_end_epoch > late_n_disc_ramp_start_epoch
        ):
            if epoch_to_train <= late_n_disc_ramp_start_epoch:
                return base_n_disc_updates
            if epoch_to_train >= late_n_disc_ramp_end_epoch:
                return late_n_disc_updates

            span = late_n_disc_ramp_end_epoch - late_n_disc_ramp_start_epoch
            elapsed = epoch_to_train - late_n_disc_ramp_start_epoch
            late_fraction = min(1.0, max(0.0, elapsed / float(span)))
            if late_n_disc_updates == base_n_disc_updates:
                return base_n_disc_updates

            phase = ((elapsed * 37) % span) / float(span)
            use_late = phase < late_fraction
            return late_n_disc_updates if use_late else base_n_disc_updates

        if late_n_disc_epoch is not None and epoch_to_train > late_n_disc_epoch:
            return late_n_disc_updates
        return base_n_disc_updates

    train_mode = "S-AIRL" if enable_safety else "GC-AIRL no-safety ablation"
    print(f"[*] 开始 {train_mode} 训练...")
    print(
        f"[*] 配置: total_epochs={total_epochs}, "
        f"eval_every={eval_freq_epochs} epoch, save_every={save_freq_epochs} epochs, "
        f"quick_eval_episodes={quick_eval_episodes}, full_eval_episodes={full_eval_episodes}, "
        f"full_eval_pre=1-{full_eval_pre_end_epoch}:{full_eval_pre_freq_epochs} epoch, "
        f"full_eval_post={full_eval_freq_epochs} epoch, epoch0_eval_episodes={epoch0_eval_episodes}"
    )
    update_generator_penalty_for_epoch(1)
    reset_generator_penalty_stats()
    
    epoch0_metrics = evaluate_policy_metrics(
        learner,
        val_dataset,
        cfg,
        n_eval_episodes=epoch0_eval_episodes,
    )
    append_eval_metrics(
        log_dir,
        {
            "epoch": 0,
            "total_timesteps": learner.num_timesteps,
            "generator_lr": base_gen_lr,
            "safety_phase": safety_phase,
            "safety_aux_weight": safety_aux_current_weight,
            "safety_cpair_additive_weight": safety_cpair_current_weight,
            "safety_late_tiny_cpair_weight": safety_late_tiny_cpair_current_weight,
            "gen_risk_penalty_lambda": gen_risk_penalty_current_lambda,
            "eval_n_episodes": epoch0_eval_episodes,
            **get_generator_penalty_stats(reset=True),
            **epoch0_metrics,
        },
    )
    print(
        "[*] Epoch 0 eval | "
        f"episodes={epoch0_eval_episodes}, "
        f"dense_norm100={epoch0_metrics['eval_dense_return_norm100']:.2f}, "
        f"paper_rank={epoch0_metrics['paper_rank_score_mean']:.2f}, "
        f"merge={epoch0_metrics['merge_success_rate']:.3f}, "
        f"endpoint={epoch0_metrics['endpoint_success_rate']:.3f}, "
        f"safety={epoch0_metrics['safety_success_rate']:.3f}, "
        f"collision={epoch0_metrics['collision_rate']:.3f}"
    )

    def build_best_key(metrics):
        return (
            -float(metrics.get("collision_rate", float("inf"))),
            float(metrics.get("safety_success_rate", float("-inf"))),
            float(metrics.get("merge_success_rate", float("-inf"))),
            float(metrics.get("endpoint_success_rate", float("-inf"))),
            float(metrics.get("eval_ep_rew_mean", float("-inf"))),
        )

    best_checkpoint_key = None
    best_checkpoint_epoch = None
    best_checkpoint_path = None

    for chunk in range(chunks):
        # ==========================================
        # 手动实现两阶段学习率退火
        # ==========================================
        chunk_start_epoch = chunk * save_freq_epochs + 1
        current_lr = base_gen_lr
        if late_gen_lr_epoch is not None and late_gen_lr is not None and chunk_start_epoch > late_gen_lr_epoch:
            current_lr = late_gen_lr
        learner.lr_schedule = lambda _: current_lr
        current_n_disc_updates = get_n_disc_updates_for_epoch(chunk_start_epoch)
        if hasattr(airl_trainer, "n_disc_updates_per_round"):
            airl_trainer.n_disc_updates_per_round = current_n_disc_updates
        update_safety_phase_for_epoch(chunk_start_epoch)
        update_safety_aux_schedule_for_epoch(chunk_start_epoch)
        update_generator_penalty_for_epoch(chunk_start_epoch)
        reset_generator_penalty_stats()
        # ==========================================
        
        last_eval_metrics = {}

        def eval_callback(round_idx):
            current_eval_epoch = chunk * save_freq_epochs + round_idx + 1
            active_full_eval_freq = full_eval_freq_epochs
            if full_eval_pre_end_epoch > 0 and current_eval_epoch <= full_eval_pre_end_epoch:
                active_full_eval_freq = full_eval_pre_freq_epochs
            is_full_eval = (current_eval_epoch % active_full_eval_freq == 0)
            eval_episodes = full_eval_episodes if is_full_eval else quick_eval_episodes
            eval_metrics = evaluate_policy_metrics(
                learner,
                val_dataset,
                cfg,
                n_eval_episodes=eval_episodes,
            )
            gen_penalty_stats = get_generator_penalty_stats(reset=True)
            eval_row = {
                "epoch": current_eval_epoch,
                "total_timesteps": learner.num_timesteps,
                "generator_lr": current_lr,
                "safety_phase": safety_phase,
                "safety_aux_weight": safety_aux_current_weight,
                "safety_cpair_additive_weight": safety_cpair_current_weight,
                "safety_late_tiny_cpair_weight": safety_late_tiny_cpair_current_weight,
                "gen_risk_penalty_lambda": gen_risk_penalty_current_lambda,
                "eval_n_episodes": eval_episodes,
                **gen_penalty_stats,
                **eval_metrics,
            }
            append_eval_metrics(log_dir, eval_row)
            last_eval_metrics.clear()
            last_eval_metrics.update(eval_metrics)
            print(
                "[*] Epoch eval | "
                f"epoch={current_eval_epoch}/{total_epochs}, "
                f"episodes={eval_episodes}, "
                f"dense_norm100={eval_metrics['eval_dense_return_norm100']:.2f}, "
                f"paper_rank={eval_metrics['paper_rank_score_mean']:.2f}, "
                f"merge={eval_metrics['merge_success_rate']:.3f}, "
                f"endpoint={eval_metrics['endpoint_success_rate']:.3f}, "
                f"safety={eval_metrics['safety_success_rate']:.3f}, "
                f"collision={eval_metrics['collision_rate']:.3f}, "
                f"gen_risk_mean={gen_penalty_stats['gen_q_safe_risk_mean']:.3f}, "
                f"gen_risk>thr={gen_penalty_stats['gen_q_safe_risk_over_threshold_rate']:.3f}, "
                f"gen_penalty_nonzero={gen_penalty_stats['gen_risk_penalty_nonzero_rate']:.3f}, "
                f"gen_penalty_mean={gen_penalty_stats['gen_risk_penalty_mean']:.6f}"
            )
            next_train_epoch = current_eval_epoch + 1
            if next_train_epoch <= total_epochs:
                update_safety_phase_for_epoch(next_train_epoch)
                update_safety_aux_schedule_for_epoch(next_train_epoch)
                update_generator_penalty_for_epoch(next_train_epoch)

        airl_trainer.train(total_timesteps=steps_per_chunk, callback=eval_callback)
        current_epoch = (chunk + 1) * save_freq_epochs

        # 在存档名中也加入架构标记
        checkpoint_path = build_safe_checkpoint_path(
            checkpoint_dir,
            run_label,
            probe_tag=probe_tag,
            prefix="baseline_policy",
            suffix="epoch",
            epoch=current_epoch,
        )
        learner.save(checkpoint_path)
        eval_metrics = last_eval_metrics
        if current_epoch >= best_select_start_epoch:
            current_best_key = build_best_key(eval_metrics)
            if best_checkpoint_key is None or current_best_key > best_checkpoint_key:
                best_checkpoint_key = current_best_key
                best_checkpoint_epoch = current_epoch
                best_checkpoint_path = build_safe_checkpoint_path(
                    checkpoint_dir,
                    run_label,
                    probe_tag=probe_tag,
                    prefix="baseline_policy",
                    suffix="best_epoch",
                    epoch=current_epoch,
                )
                learner.save(best_checkpoint_path)
                print(f"[*] Best checkpoint updated -> {best_checkpoint_path}")

        print(
            f"[*] 进度 {current_epoch}/{total_epochs} Epochs -> 模型已保存至: {checkpoint_path}"
        )
        print(
            "[*] 周期评估 | "
            f"merge_success={eval_metrics['merge_success_rate']:.3f}, "
            f"endpoint_success={eval_metrics['endpoint_success_rate']:.3f}, "
            f"safety_success={eval_metrics['safety_success_rate']:.3f}, "
            f"mean_speed={eval_metrics['mean_speed']:.2f}, "
            f"max_speed={eval_metrics['max_speed']:.2f}, "
            f"mean_abs_acc={eval_metrics['mean_abs_acc']:.2f}, "
            f"mean_abs_jerk={eval_metrics['mean_abs_jerk']:.2f}, "
            f"min_ttc={eval_metrics['min_ttc']:.2f}, "
            f"min_thw={eval_metrics['min_thw']:.2f}, "
            f"eval_ep_rew={eval_metrics['eval_ep_rew_mean']:.3f}, "
            f"dense_norm100={eval_metrics['eval_dense_return_norm100']:.2f}, "
            f"paper_rank={eval_metrics['paper_rank_score_mean']:.2f}, "
            f"lr={current_lr:.6f}, "
            f"n_disc={current_n_disc_updates}"
        )

    # ==========================================
    # 5. 保存最终模型
    # ==========================================
    final_checkpoint_path = build_safe_checkpoint_path(
        checkpoint_dir,
        run_label,
        probe_tag=probe_tag,
        prefix="airl_policy_baseline",
        suffix="final",
    )
    learner.save(final_checkpoint_path)
    print(f"\n训练全部完成，最终策略已保存为 {final_checkpoint_path}。")

    if best_checkpoint_path is not None:
        print(f"[*] Best checkpoint in selection window: epoch={best_checkpoint_epoch}, path={best_checkpoint_path}")

if __name__ == "__main__":
    main()
