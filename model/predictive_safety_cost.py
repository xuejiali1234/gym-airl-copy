from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class PredictiveSafetyCostNetwork(nn.Module):
    """Action-aware predictive safety network with risk and critical heads."""

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
        self.feature_dim = self.hidden_dim
        self._grad_hooks = []

        input_dim = self.state_dim + (self.action_dim if self.use_action else 0)
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.risk_head = nn.Linear(hidden_dim, 1)
        self.critical_head = nn.Linear(hidden_dim, 1)

    def _prepare_inputs(self, state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        core_state = state[..., : self.state_dim]
        if self.use_action:
            if action is None:
                raise ValueError("PredictiveSafetyCostNetwork requires action when use_action=True.")
            return torch.cat([core_state, action], dim=-1)
        return core_state

    def forward(self, state: torch.Tensor, action: Optional[torch.Tensor] = None, *, return_feature: bool = False):
        x = self._prepare_inputs(state, action)
        feat = self.feature_net(x)
        risk_logit = self.risk_head(feat)
        if return_feature:
            return risk_logit, feat
        return risk_logit

    def forward_heads(
        self,
        state: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        *,
        return_feature: bool = False,
    ):
        x = self._prepare_inputs(state, action)
        feat = self.feature_net(x)
        risk_logit = self.risk_head(feat)
        critical_logit = self.critical_head(feat)
        if return_feature:
            return risk_logit, critical_logit, feat
        return risk_logit, critical_logit

    def get_feature(self, state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self._prepare_inputs(state, action)
        return self.feature_net(x)

    def get_safety_risk(self, state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        return torch.sigmoid(self.forward(state, action))

    def get_critical_prob(self, state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        _, critical_logit = self.forward_heads(state, action)
        return torch.sigmoid(critical_logit)

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

        last_linear = self.feature_net[2]
        for p in last_linear.parameters():
            p.requires_grad_(True)
        for p in self.risk_head.parameters():
            p.requires_grad_(True)
        for p in self.critical_head.parameters():
            p.requires_grad_(True)

        self.train()

        if grad_scale < 1.0:
            for param in self.parameters():
                if param.requires_grad:
                    self._grad_hooks.append(param.register_hook(lambda grad, scale=grad_scale: grad * scale))
