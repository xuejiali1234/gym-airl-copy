import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from imitation.rewards.reward_nets import PredictProcessedWrapper, RewardNet

# ==========================================
# 1. 核心注意力层 (Social Attention Layer)
# ==========================================
class SocialAttentionLayer(nn.Module):
    """
    轻量级社会注意力机制 (带缺失车辆 Mask 处理)
    """
    def __init__(self, obs_dim, hidden_dim=64, num_heads=4):
        super(SocialAttentionLayer, self).__init__()
        ego_dim = 4
        nbr_dim = 4
        
        self.query_proj = nn.Linear(ego_dim, hidden_dim)
        self.key_proj = nn.Linear(nbr_dim, hidden_dim)
        self.val_proj = nn.Linear(nbr_dim, hidden_dim)
        
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        
    def forward(self, state):
        ego = state[:, 0:4]
            
        # 提取周车特征 [Batch, 3, 4]
        nbrs = state[:, 4:16].view(-1, 3, 4) 
        
        # [修复重点] 计算 Mask：如果某辆车的 4 个特征全接近 0，说明该位置没车
        # key_padding_mask 要求 True 代表需要被忽略 (Masked)
        key_padding_mask = (nbrs.abs().sum(dim=-1) < 1e-5)
        
        # 防止极端情况：如果周围一辆车都没有，所有的 key 都被 mask，Softmax 会报 NaN
        # 我们检测每一行，如果全 True，就强制把第一个位置放开 (设为 False)
        all_masked = key_padding_mask.all(dim=1)
        if all_masked.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_masked, 0] = False
        
        Q = self.query_proj(ego).unsqueeze(1)  
        K = self.key_proj(nbrs)                
        V = self.val_proj(nbrs)                
        
        # 传入 key_padding_mask，注意力机制会自动忽略没车的位置
        attn_out, _ = self.attn(Q, K, V, key_padding_mask=key_padding_mask)       
        
        out_feat = attn_out.squeeze(1)
        
        # [修复重点] 如果周围真的没车，输出特征强制归零，消除背景噪声
        if all_masked.any():
            mask_expanded = (~all_masked).unsqueeze(1).float()
            out_feat = out_feat * mask_expanded

        return out_feat


# ==========================================
# 2. PPO 特征提取器
# ==========================================
class AttentionFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, hidden_dim=64):
        obs_dim = observation_space.shape[0]
        features_dim = obs_dim + hidden_dim
        super(AttentionFeaturesExtractor, self).__init__(observation_space, features_dim)
        
        self.attention = SocialAttentionLayer(obs_dim, hidden_dim)

    def forward(self, observations):
        feat = self.attention(observations)
        return torch.cat([observations, feat], dim=-1)


class GoalConditionedMLPFeaturesExtractor(BaseFeaturesExtractor):
    """Structured MLP extractor for the goal-only baseline without attention."""

    def __init__(
        self,
        observation_space,
        state_dim=16,
        goal_dim=2,
        state_hidden_dim=64,
        goal_hidden_dim=32,
    ):
        features_dim = state_hidden_dim + goal_hidden_dim
        super().__init__(observation_space, features_dim)
        self.state_dim = state_dim
        self.goal_dim = goal_dim

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, state_hidden_dim),
            nn.Tanh(),
            nn.Linear(state_hidden_dim, state_hidden_dim),
            nn.Tanh(),
        )
        self.goal_encoder = nn.Sequential(
            nn.Linear(goal_dim, goal_hidden_dim),
            nn.Tanh(),
            nn.Linear(goal_hidden_dim, goal_hidden_dim),
            nn.Tanh(),
        )

    def forward(self, observations):
        state = observations[:, : self.state_dim]
        goal = observations[:, self.state_dim : self.state_dim + self.goal_dim]
        state_feat = self.state_encoder(state)
        goal_feat = self.goal_encoder(goal)
        return torch.cat([state_feat, goal_feat], dim=-1)


# ==========================================
# 3. AIRL 奖励网络 (修复张量通道)
# ==========================================
class AttentionRewardNet(RewardNet):
    def __init__(self, observation_space, action_space, hidden_dim=64):
        super(AttentionRewardNet, self).__init__(observation_space, action_space)
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        
        self.attention = SocialAttentionLayer(obs_dim, hidden_dim)
        
        # 【严格对齐基线】：将下游 MLP 的深度和宽度对齐为 (128, 128)
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim + hidden_dim + act_dim, 128), # state + attention + action
            nn.Tanh(),
            nn.Linear(128, 128),                  # 第二层对齐 128
            nn.Tanh(),
            nn.Linear(128, 1)                     # 输出 1 维标量
        )

    def forward(self, state, action, next_state, done):
        feat = self.attention(state)
        x = torch.cat([state, feat, action], dim=-1)
        reward = self.mlp(x)
        return reward.squeeze(-1)


class GoalRewardWrapper(PredictProcessedWrapper):
    """Adds the one-time merge-goal bonus and optional late high-risk penalty to policy reward."""

    def __init__(
        self,
        base,
        expert_mean_x,
        expert_std_x,
        divider_x,
        goal_bonus=0.5,
        high_risk_penalty_lambda=0.0,
        high_risk_threshold=0.7,
        high_risk_penalty_clip_min=-0.03,
        high_risk_penalty_clip_max=0.0,
        high_risk_source="q_safe_risk",
        predictive_safety_oracle=None,
    ):
        super().__init__(base)
        self.expert_mean_x = float(expert_mean_x)
        self.expert_std_x = float(expert_std_x)
        self.target_x_threshold = float(divider_x - 3.28)
        self.goal_bonus = float(goal_bonus)
        self.high_risk_penalty_lambda = float(high_risk_penalty_lambda)
        self.high_risk_threshold = float(high_risk_threshold)
        self.high_risk_penalty_clip_min = float(high_risk_penalty_clip_min)
        self.high_risk_penalty_clip_max = float(high_risk_penalty_clip_max)
        self.high_risk_source = str(high_risk_source)
        self.predictive_safety_oracle = predictive_safety_oracle
        self._already_rewarded = None
        self.reset_high_risk_penalty_stats()

    def _denormalize_x(self, x):
        return x * self.expert_std_x + self.expert_mean_x

    def set_high_risk_penalty(
        self,
        *,
        lambda_risk=None,
        threshold=None,
        clip_min=None,
        clip_max=None,
        source=None,
    ):
        if lambda_risk is not None:
            self.high_risk_penalty_lambda = float(lambda_risk)
        if threshold is not None:
            self.high_risk_threshold = float(threshold)
        if clip_min is not None:
            self.high_risk_penalty_clip_min = float(clip_min)
        if clip_max is not None:
            self.high_risk_penalty_clip_max = float(clip_max)
        if source is not None:
            self.high_risk_source = str(source)

    def reset_high_risk_penalty_stats(self):
        self._high_risk_penalty_stats = {
            "calls": 0,
            "samples": 0,
            "risk_sum": 0.0,
            "risk_max": 0.0,
            "risk_over_threshold": 0,
            "penalty_sum": 0.0,
            "penalty_abs_sum": 0.0,
            "penalty_min": 0.0,
            "penalty_nonzero": 0,
            "penalty_clip_min": 0,
        }

    def _record_high_risk_penalty_stats(self, risk_score, risk_penalty):
        risk_arr = np.asarray(risk_score, dtype=np.float64).reshape(-1)
        penalty_arr = np.asarray(risk_penalty, dtype=np.float64).reshape(-1)
        if risk_arr.size == 0:
            return

        if penalty_arr.size == 1 and risk_arr.size != 1:
            penalty_arr = np.full_like(risk_arr, float(penalty_arr[0]), dtype=np.float64)

        stats = self._high_risk_penalty_stats
        stats["calls"] += 1
        stats["samples"] += int(risk_arr.size)
        stats["risk_sum"] += float(np.sum(risk_arr))
        stats["risk_max"] = max(float(stats["risk_max"]), float(np.max(risk_arr)))
        stats["risk_over_threshold"] += int(np.sum(risk_arr > self.high_risk_threshold))
        stats["penalty_sum"] += float(np.sum(penalty_arr))
        stats["penalty_abs_sum"] += float(np.sum(np.abs(penalty_arr)))
        stats["penalty_min"] = min(float(stats["penalty_min"]), float(np.min(penalty_arr)))
        stats["penalty_nonzero"] += int(np.sum(np.abs(penalty_arr) > 1e-12))
        stats["penalty_clip_min"] += int(
            np.sum(penalty_arr <= self.high_risk_penalty_clip_min + 1e-12)
        )

    def get_high_risk_penalty_stats(self, reset=False):
        stats = self._high_risk_penalty_stats
        samples = int(stats["samples"])
        if samples > 0:
            result = {
                "gen_risk_penalty_calls": int(stats["calls"]),
                "gen_risk_penalty_samples": samples,
                "gen_q_safe_risk_mean": float(stats["risk_sum"]) / samples,
                "gen_q_safe_risk_max": float(stats["risk_max"]),
                "gen_q_safe_risk_over_threshold_rate": float(stats["risk_over_threshold"]) / samples,
                "gen_risk_penalty_mean": float(stats["penalty_sum"]) / samples,
                "gen_risk_penalty_abs_mean": float(stats["penalty_abs_sum"]) / samples,
                "gen_risk_penalty_min": float(stats["penalty_min"]),
                "gen_risk_penalty_nonzero_rate": float(stats["penalty_nonzero"]) / samples,
                "gen_risk_penalty_clip_min_rate": float(stats["penalty_clip_min"]) / samples,
            }
        else:
            result = {
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
            }

        result["gen_risk_penalty_active_lambda"] = float(self.high_risk_penalty_lambda)
        result["gen_risk_penalty_active_threshold"] = float(self.high_risk_threshold)
        result["gen_risk_penalty_source"] = str(self.high_risk_source)

        if reset:
            self.reset_high_risk_penalty_stats()
        return result

    def _get_high_risk_score(self, safety_debug, state_tensor, action_tensor):
        if self.high_risk_source in {"target_lead", "target_lane_lead"} and self.predictive_safety_oracle is not None:
            oracle_debug = self.predictive_safety_oracle.analyze_batch(state_tensor, action_tensor)
            return oracle_debug["target_lead_risk"].squeeze(-1).detach().cpu().numpy()
        return safety_debug["q_safe_risk"].squeeze(-1).detach().cpu().numpy()

    def predict_processed(self, state, action, next_state, done, **kwargs):
        base_reward = self.base.predict_processed(
            state,
            action,
            next_state,
            done,
            **kwargs,
        )

        prev_x = self._denormalize_x(np.asarray(state)[..., 0])
        next_x = self._denormalize_x(np.asarray(next_state)[..., 0])
        crossed_into_target_lane = (
            (prev_x >= self.target_x_threshold) & (next_x < self.target_x_threshold)
        )

        crossed_into_target_lane = np.asarray(crossed_into_target_lane, dtype=bool).reshape(-1)
        done_arr = np.asarray(done, dtype=bool).reshape(-1)

        if self._already_rewarded is None or self._already_rewarded.shape != crossed_into_target_lane.shape:
            self._already_rewarded = np.zeros_like(crossed_into_target_lane, dtype=bool)

        give_bonus = crossed_into_target_lane & (~self._already_rewarded)
        goal_reward = give_bonus.astype(np.float32) * self.goal_bonus

        self._already_rewarded = self._already_rewarded | give_bonus
        self._already_rewarded[done_arr] = False

        if np.asarray(base_reward).ndim == 0:
            goal_reward = goal_reward.reshape(())

        total_reward = base_reward + goal_reward

        if self.high_risk_penalty_lambda > 0.0 and hasattr(self.base, "get_safety_debug"):
            try:
                base_device = next(self.base.parameters()).device
            except StopIteration:
                base_device = torch.device("cpu")

            state_tensor = torch.as_tensor(np.asarray(state), dtype=torch.float32, device=base_device)
            action_tensor = torch.as_tensor(np.asarray(action), dtype=torch.float32, device=base_device)
            with torch.no_grad():
                safety_debug = self.base.get_safety_debug(state_tensor, action_tensor)
                risk_score = self._get_high_risk_score(safety_debug, state_tensor, action_tensor)

            denom = max(1e-6, 1.0 - self.high_risk_threshold)
            risk_excess = np.clip((risk_score - self.high_risk_threshold) / denom, 0.0, 1.0)
            risk_penalty = -self.high_risk_penalty_lambda * risk_excess
            risk_penalty = np.clip(
                risk_penalty,
                self.high_risk_penalty_clip_min,
                self.high_risk_penalty_clip_max,
            )
            self._record_high_risk_penalty_stats(risk_score, risk_penalty)
            if np.asarray(total_reward).ndim == 0:
                risk_penalty = np.asarray(risk_penalty).reshape(())
            total_reward = total_reward + risk_penalty

        return total_reward
