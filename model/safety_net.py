from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from imitation.rewards.reward_nets import RewardNet

from model.attention_net import SocialAttentionLayer


class SafetyNetwork(nn.Module):
    """
    Pretrained S-Net.

    By default it consumes the 16-D core driving state (goal dims are ignored).
    It outputs:
    - a safety logit for BCE training
    - an intermediate feature vector for feature-level fusion into the AIRL reward net
    """

    def __init__(
        self,
        state_dim: int = 16,
        hidden_dim: int = 128,
        action_dim: int = 0,
        use_action: bool = False,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.action_dim = int(action_dim)
        self.use_action = bool(use_action and action_dim > 0)

        input_dim = self.state_dim + (self.action_dim if self.use_action else 0)
        self.feature_dim = self.hidden_dim

        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.head = nn.Linear(hidden_dim, 1)

    def _prepare_inputs(self, state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        core_state = state[..., : self.state_dim]
        if self.use_action:
            if action is None:
                raise ValueError("SafetyNetwork was created with use_action=True, but action is None.")
            return torch.cat([core_state, action], dim=-1)
        return core_state

    def forward(
        self,
        state: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        *,
        return_feature: bool = False,
    ):
        x = self._prepare_inputs(state, action)
        feat = self.feature_net(x)
        logits = self.head(feat)
        if return_feature:
            return logits, feat
        return logits

    def get_feature(self, state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self._prepare_inputs(state, action)
        return self.feature_net(x)

    def get_safety_risk(self, state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        return torch.sigmoid(self.forward(state, action))


def freeze_module(module: nn.Module) -> None:
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)


class SafeAttentionRewardNet(RewardNet):
    """
    Current-codebase-compatible AIRL reward net with safety fusion.

    Fusion structure:
      behavior branch: [state, attention(state), action] -> MLP -> behavior_feat
      safety branch:   S-Net(state[,action]) -> [feature, risk] -> MLP -> safety_feat
      final reward:    concat(behavior_feat, safety_feat) -> MLP -> scalar reward

    This matches your old design idea of concatenating safety information into the
    discriminator/reward path, while staying compatible with `imitation`'s RewardNet.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        safety_net: SafetyNetwork,
        *,
        hidden_dim: int = 64,
        safety_embed_dim: int = 32,
        freeze_safety: bool = True,
    ) -> None:
        super().__init__(observation_space, action_space)
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]

        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.core_state_dim = 16
        self.freeze_safety = bool(freeze_safety)

        self.attention = SocialAttentionLayer(obs_dim, hidden_dim)
        self.behavior_mlp = nn.Sequential(
            nn.Linear(obs_dim + hidden_dim + act_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )

        self.safety_net = safety_net
        if self.freeze_safety:
            freeze_module(self.safety_net)

        safety_input_dim = self.safety_net.feature_dim + 1
        self.safety_mlp = nn.Sequential(
            nn.Linear(safety_input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, safety_embed_dim),
            nn.Tanh(),
        )

        self.head = nn.Sequential(
            nn.Linear(128 + safety_embed_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def _get_safety_embedding(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        safety_action = action if self.safety_net.use_action else None
        core_state = state[..., : self.core_state_dim]

        if self.freeze_safety:
            with torch.no_grad():
                safety_feat = self.safety_net.get_feature(core_state, safety_action)
                safety_risk = self.safety_net.get_safety_risk(core_state, safety_action)
        else:
            safety_feat = self.safety_net.get_feature(core_state, safety_action)
            safety_risk = self.safety_net.get_safety_risk(core_state, safety_action)

        safety_input = torch.cat([safety_feat, safety_risk], dim=-1)
        return self.safety_mlp(safety_input)

    def forward(self, state, action, next_state, done):
        attn_feat = self.attention(state)
        behavior_input = torch.cat([state, attn_feat, action], dim=-1)
        behavior_feat = self.behavior_mlp(behavior_input)

        safety_feat = self._get_safety_embedding(state, action)
        fused_feat = torch.cat([behavior_feat, safety_feat], dim=-1)
        reward = self.head(fused_feat)
        return reward.squeeze(-1)


class SafeMLPRewardNet(RewardNet):
    """MLP baseline version of the same safety-fusion idea."""

    def __init__(
        self,
        observation_space,
        action_space,
        safety_net: SafetyNetwork,
        *,
        safety_embed_dim: int = 32,
        freeze_safety: bool = True,
    ) -> None:
        super().__init__(observation_space, action_space)
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]

        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.core_state_dim = 16
        self.freeze_safety = bool(freeze_safety)

        self.behavior_mlp = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )

        self.safety_net = safety_net
        if self.freeze_safety:
            freeze_module(self.safety_net)

        safety_input_dim = self.safety_net.feature_dim + 1
        self.safety_mlp = nn.Sequential(
            nn.Linear(safety_input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, safety_embed_dim),
            nn.Tanh(),
        )

        self.head = nn.Sequential(
            nn.Linear(128 + safety_embed_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def _get_safety_embedding(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        safety_action = action if self.safety_net.use_action else None
        core_state = state[..., : self.core_state_dim]

        if self.freeze_safety:
            with torch.no_grad():
                safety_feat = self.safety_net.get_feature(core_state, safety_action)
                safety_risk = self.safety_net.get_safety_risk(core_state, safety_action)
        else:
            safety_feat = self.safety_net.get_feature(core_state, safety_action)
            safety_risk = self.safety_net.get_safety_risk(core_state, safety_action)

        safety_input = torch.cat([safety_feat, safety_risk], dim=-1)
        return self.safety_mlp(safety_input)

    def forward(self, state, action, next_state, done):
        behavior_feat = self.behavior_mlp(torch.cat([state, action], dim=-1))
        safety_feat = self._get_safety_embedding(state, action)
        fused_feat = torch.cat([behavior_feat, safety_feat], dim=-1)
        reward = self.head(fused_feat)
        return reward.squeeze(-1)
