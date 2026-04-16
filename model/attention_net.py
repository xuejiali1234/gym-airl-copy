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
    """Adds the one-time merge-goal bonus only to policy training reward."""

    def __init__(self, base, expert_mean_x, expert_std_x, divider_x, goal_bonus=0.5):
        super().__init__(base)
        self.expert_mean_x = float(expert_mean_x)
        self.expert_std_x = float(expert_std_x)
        self.target_x_threshold = float(divider_x - 3.28)
        self.goal_bonus = float(goal_bonus)
        self._already_rewarded = None

    def _denormalize_x(self, x):
        return x * self.expert_std_x + self.expert_mean_x

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

        return base_reward + goal_reward
