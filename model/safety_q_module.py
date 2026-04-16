from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from imitation.rewards.reward_nets import RewardNet

from model.attention_net import SocialAttentionLayer


class SafetyQNetwork(nn.Module):
    """
    Action-aware safety network approximating Q_safe(s, a).

    Inputs:
      - normalized core driving state of shape [..., 16]
      - normalized action of shape [..., 2] (optional if use_action=False)

    Outputs:
      - safety logit (higher => more unsafe)
      - hidden safety feature for discriminator fusion
    """

    def __init__(
        self,
        state_dim: int = 16,
        action_dim: int = 2,
        hidden_dim: int = 128,
        use_action: bool = True,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.use_action = bool(use_action)
        self._grad_hooks = []

        input_dim = self.state_dim + (self.action_dim if self.use_action else 0)
        self.feature_dim = self.hidden_dim

        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.head = nn.Linear(hidden_dim, 1)

    def _prepare_inputs(
        self,
        state: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        core_state = state[..., : self.state_dim]
        if self.use_action:
            if action is None:
                raise ValueError("SafetyQNetwork requires action when use_action=True.")
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

    def _clear_grad_hooks(self) -> None:
        for hook in self._grad_hooks:
            hook.remove()
        self._grad_hooks = []

    def freeze_all(self) -> None:
        self._clear_grad_hooks()
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def enable_light_finetune(self, grad_scale: float = 1.0) -> None:
        self._clear_grad_hooks()
        for p in self.parameters():
            p.requires_grad_(False)

        # Only unfreeze the last feature layer and the safety head.
        last_linear = self.feature_net[2]
        for p in last_linear.parameters():
            p.requires_grad_(True)
        for p in self.head.parameters():
            p.requires_grad_(True)

        self.train()

        if grad_scale < 1.0:
            for param in self.parameters():
                if param.requires_grad:
                    self._grad_hooks.append(param.register_hook(lambda grad, scale=grad_scale: grad * scale))


class ZeroSafetyQNetwork(nn.Module):
    """Zero-valued safety prior for structure-preserving safety ablations."""

    def __init__(
        self,
        state_dim: int = 16,
        action_dim: int = 2,
        feature_dim: int = 128,
        use_action: bool = True,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.feature_dim = int(feature_dim)
        self.use_action = bool(use_action)

    def forward(
        self,
        state: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        *,
        return_feature: bool = False,
    ):
        batch_shape = state.shape[:-1]
        logit = state.new_zeros(*batch_shape, 1)
        if return_feature:
            feat = state.new_zeros(*batch_shape, self.feature_dim)
            return logit, feat
        return logit

    def get_feature(self, state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        return state.new_zeros(*state.shape[:-1], self.feature_dim)

    def get_safety_risk(self, state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        return torch.sigmoid(self.forward(state, action))

    def freeze_all(self) -> None:
        self.eval()

    def enable_light_finetune(self, grad_scale: float = 1.0) -> None:
        self.eval()


def freeze_module(module: nn.Module) -> None:
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)


class _SafetyFusionMixin:
    """Shared utilities for safety-aware reward networks."""

    def _init_safety(
        self,
        safety_net: SafetyQNetwork,
        *,
        safety_embed_dim: int = 32,
        freeze_safety: bool = True,
        fuse_safety_feature: bool = True,
    ) -> None:
        self.core_state_dim = 16
        self.safety_net = safety_net
        self.freeze_safety = bool(freeze_safety)
        self.fuse_safety_feature = bool(fuse_safety_feature)
        self.safety_embed_dim = 1 if not self.fuse_safety_feature else int(safety_embed_dim)

        if self.freeze_safety:
            freeze_module(self.safety_net)

        safety_input_dim = 1
        if self.fuse_safety_feature:
            safety_input_dim += self.safety_net.feature_dim
            self.safety_proj = nn.Sequential(
                nn.Linear(safety_input_dim, 64),
                nn.Tanh(),
                nn.Linear(64, self.safety_embed_dim),
                nn.Tanh(),
            )
        else:
            self.safety_proj = nn.Identity()

    def _get_safety_outputs(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        core_state = state[..., : self.core_state_dim]
        safety_action = action if self.safety_net.use_action else None
        if self.freeze_safety:
            with torch.no_grad():
                q_logit, q_feat = self.safety_net(core_state, safety_action, return_feature=True)
        else:
            q_logit, q_feat = self.safety_net(core_state, safety_action, return_feature=True)
        q_risk = torch.sigmoid(q_logit)
        return q_logit, q_risk, q_feat

    def _get_safety_embedding(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q_logit, _, q_feat = self._get_safety_outputs(state, action)
        pieces = [q_logit]
        if self.fuse_safety_feature:
            pieces = [q_feat, q_logit]
        safety_input = torch.cat(pieces, dim=-1)
        return self.safety_proj(safety_input)

    def set_safety_training_phase(self, phase: str, grad_scale: float = 1.0) -> None:
        if phase == "frozen":
            self.freeze_safety = True
            self.safety_net.freeze_all()
            return
        if phase == "light_unfreeze":
            self.freeze_safety = False
            self.safety_net.enable_light_finetune(grad_scale=grad_scale)
            return
        raise ValueError(f"Unknown safety training phase: {phase}")

    def disc_aux_safety_loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        disc_logits: torch.Tensor,
        labels_expert_is_one: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mild discriminator intervention.

        Only penalize *generated* risky samples that the discriminator still scores as
        expert-like. This keeps the intervention light and avoids fighting expert labels.
        """
        _, q_risk, _ = self._get_safety_outputs(state, action)
        q_risk = q_risk.squeeze(-1)

        gen_mask = (labels_expert_is_one < 0.5).float()
        risky_gen_weight = q_risk * gen_mask
        denom = risky_gen_weight.sum().clamp_min(1.0)

        # softplus(logit) is large when the discriminator thinks the sample is expert-like.
        aux = (F.softplus(disc_logits) * risky_gen_weight).sum() / denom
        return aux

    def get_safety_debug(self, state: torch.Tensor, action: torch.Tensor) -> dict[str, torch.Tensor]:
        q_logit, q_risk, _ = self._get_safety_outputs(state, action)
        return {
            "q_safe_logit": q_logit.detach(),
            "q_safe_risk": q_risk.detach(),
        }


class SafeQAttentionRewardNet(_SafetyFusionMixin, RewardNet):
    """
    Reward / discriminator network matching your diagram:

      (s, a) -> behavior branch (attention + MLP)
      (s, a) -> trained S-Net -> q_safe(s,a) + safety feature -> MLP
      concat -> MLP -> scalar discriminator reward/logit component

    Compared with your current SafeAttentionRewardNet, this version is *action-aware*
    on the safety side and exposes an auxiliary discriminator safety loss hook.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        safety_net: SafetyQNetwork,
        *,
        hidden_dim: int = 64,
        safety_embed_dim: int = 32,
        freeze_safety: bool = True,
        fuse_safety_feature: bool = True,
    ) -> None:
        super().__init__(observation_space, action_space)
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]

        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)

        self.attention = SocialAttentionLayer(obs_dim, hidden_dim)
        self.behavior_proj = nn.Sequential(
            nn.Linear(obs_dim + hidden_dim + act_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )

        self._init_safety(
            safety_net,
            safety_embed_dim=safety_embed_dim,
            freeze_safety=freeze_safety,
            fuse_safety_feature=fuse_safety_feature,
        )

        self.head = nn.Sequential(
            nn.Linear(128 + self.safety_embed_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, state, action, next_state, done):
        attn_feat = self.attention(state)
        behavior_feat = self.behavior_proj(torch.cat([state, attn_feat, action], dim=-1))
        safety_feat = self._get_safety_embedding(state, action)
        fused = torch.cat([behavior_feat, safety_feat], dim=-1)
        reward = self.head(fused)
        return reward.squeeze(-1)


class SafeQMLPRewardNet(_SafetyFusionMixin, RewardNet):
    """MLP baseline version of the same safety-fusion idea."""

    def __init__(
        self,
        observation_space,
        action_space,
        safety_net: SafetyQNetwork,
        *,
        safety_embed_dim: int = 32,
        freeze_safety: bool = True,
        fuse_safety_feature: bool = True,
    ) -> None:
        super().__init__(observation_space, action_space)
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]

        self.behavior_proj = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )

        self._init_safety(
            safety_net,
            safety_embed_dim=safety_embed_dim,
            freeze_safety=freeze_safety,
            fuse_safety_feature=fuse_safety_feature,
        )

        self.head = nn.Sequential(
            nn.Linear(128 + self.safety_embed_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, state, action, next_state, done):
        behavior_feat = self.behavior_proj(torch.cat([state, action], dim=-1))
        safety_feat = self._get_safety_embedding(state, action)
        fused = torch.cat([behavior_feat, safety_feat], dim=-1)
        reward = self.head(fused)
        return reward.squeeze(-1)
