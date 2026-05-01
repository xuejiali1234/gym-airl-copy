from __future__ import annotations

from typing import Mapping, Optional

import torch as th
from torch.nn import functional as F

from imitation.algorithms.adversarial import common
from imitation.algorithms.adversarial.airl import AIRL
from imitation.rewards import reward_nets


class MildSafetyAIRL(AIRL):
    """AIRL with a mild auxiliary safety loss on discriminator updates.

    The standard AIRL discriminator loss stays unchanged. We only add a small
    extra term when the *base* reward net exposes `disc_aux_safety_loss(...)`.
    This is exactly the "light intervention" version: do not redesign the whole
    discriminator objective, just nudge it away from scoring risky generator
    transitions as expert-like.
    """

    def __init__(
        self,
        *args,
        safety_loss_weight: float = 0.05,
        safety_reg_mode: str = "legacy",
        safety_reg_margin: float = 0.2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.safety_loss_weight = float(safety_loss_weight)
        self.safety_reg_mode = str(safety_reg_mode)
        self.safety_reg_margin = float(safety_reg_margin)

    @staticmethod
    def _build_predictive_candidate_actions(action: th.Tensor) -> th.Tensor:
        """Construct same-state candidate actions for predictive safety ranking."""
        lateral = action[:, 0]
        longitudinal = action[:, 1]

        hold_lane = th.stack([th.zeros_like(lateral), longitudinal], dim=-1)
        delay_merge = th.stack([0.5 * lateral.clamp(max=0.0), longitudinal], dim=-1)
        mild_decel = th.stack([lateral, (longitudinal - 0.25).clamp(-1.0, 1.0)], dim=-1)
        mild_accel = th.stack([lateral, (longitudinal + 0.25).clamp(-1.0, 1.0)], dim=-1)
        mild_merge = th.stack([(lateral - 0.25).clamp(-1.0, 1.0), longitudinal], dim=-1)

        candidates = th.stack(
            [
                action,
                hold_lane,
                delay_merge,
                mild_decel,
                mild_accel,
                mild_merge,
            ],
            dim=1,
        )
        return candidates.clamp(-1.0, 1.0)

    def _predictive_candidate_pair_loss(
        self,
        *,
        base_reward_net,
        batch,
    ) -> th.Tensor:
        labels = batch["labels_expert_is_one"] < 0.5
        if not labels.any():
            return th.zeros((), device=batch["state"].device)

        state = batch["state"][labels]
        action = batch["action"][labels]
        next_state = batch["next_state"][labels]
        done = batch["done"][labels]
        num_samples = state.shape[0]

        candidate_actions = self._build_predictive_candidate_actions(action)
        num_candidates = candidate_actions.shape[1]

        state_rep = state.unsqueeze(1).expand(-1, num_candidates, -1).reshape(-1, state.shape[-1])
        next_state_rep = next_state.unsqueeze(1).expand(-1, num_candidates, -1).reshape(-1, next_state.shape[-1])
        done_rep = done.unsqueeze(1).expand(-1, num_candidates).reshape(-1)
        candidate_actions_flat = candidate_actions.reshape(-1, action.shape[-1])

        safety_debug = base_reward_net.get_safety_debug(state_rep, candidate_actions_flat)
        candidate_risk = safety_debug["q_safe_risk"].reshape(num_samples, num_candidates)

        candidate_reward = self._reward_net(
            state_rep,
            candidate_actions_flat,
            next_state_rep,
            done_rep,
        ).reshape(num_samples, num_candidates)

        safe_idx = th.argmin(candidate_risk.squeeze(-1), dim=1)
        max_risk_idx = th.argmax(candidate_risk.squeeze(-1), dim=1)
        policy_risk = candidate_risk[:, 0, 0]
        risky_idx = th.where(policy_risk >= 0.7, th.zeros_like(max_risk_idx), max_risk_idx)

        row_idx = th.arange(num_samples, device=state.device)
        safe_reward = candidate_reward[row_idx, safe_idx]
        risky_reward = candidate_reward[row_idx, risky_idx]
        safe_risk = candidate_risk[row_idx, safe_idx, 0]
        risky_risk = candidate_risk[row_idx, risky_idx, 0]

        risk_gap = (risky_risk - safe_risk).detach()
        valid_mask = risk_gap > 1e-4
        if not valid_mask.any():
            return th.zeros((), device=state.device)

        pair_loss = F.relu(self.safety_reg_margin + risky_reward - safe_reward)
        weighted_loss = pair_loss[valid_mask] * risk_gap[valid_mask]
        return weighted_loss.mean()

    def _unwrap_reward_net(self):
        reward_net = self._reward_net
        while isinstance(reward_net, reward_nets.RewardNetWrapper):
            reward_net = reward_net.base
        return reward_net

    def train_disc(
        self,
        *,
        expert_samples: Optional[Mapping] = None,
        gen_samples: Optional[Mapping] = None,
    ) -> Mapping[str, float]:
        with self.logger.accumulate_means("disc"):
            self._disc_opt.zero_grad()
            batch_iter = self._make_disc_train_batches(
                gen_samples=gen_samples,
                expert_samples=expert_samples,
            )

            base_reward_net = self._unwrap_reward_net()
            last_logits = None
            last_labels = None
            last_bce = None
            last_total = None
            last_aux = None
            last_reg = None
            last_reward_expert = None
            last_reward_gen = None
            last_reward_gap = None

            for batch in batch_iter:
                disc_logits = self.logits_expert_is_high(
                    batch["state"],
                    batch["action"],
                    batch["next_state"],
                    batch["done"],
                    batch["log_policy_act_prob"],
                )

                bce_loss = F.binary_cross_entropy_with_logits(
                    disc_logits,
                    batch["labels_expert_is_one"].float(),
                )

                raw_reward = self._reward_net(
                    batch["state"],
                    batch["action"],
                    batch["next_state"],
                    batch["done"],
                )

                aux_loss = th.zeros((), device=disc_logits.device)
                if self.safety_loss_weight > 0.0:
                    if self.safety_reg_mode == "predictive_ranking" and hasattr(base_reward_net, "get_safety_debug"):
                        safety_debug = base_reward_net.get_safety_debug(
                            batch["state"],
                            batch["action"],
                        )
                        q_risk = safety_debug["q_safe_risk"].squeeze(-1)
                        safe_mask = q_risk < 0.2
                        risky_mask = q_risk > 0.7
                        if safe_mask.any() and risky_mask.any():
                            aux_loss = th.relu(
                                self.safety_reg_margin
                                + raw_reward[risky_mask].mean()
                                - raw_reward[safe_mask].mean()
                            )
                    elif self.safety_reg_mode == "predictive_candidate_ranking" and hasattr(base_reward_net, "get_safety_debug"):
                        aux_loss = self._predictive_candidate_pair_loss(
                            base_reward_net=base_reward_net,
                            batch=batch,
                        )
                    elif hasattr(base_reward_net, "disc_aux_safety_loss"):
                        aux_loss = base_reward_net.disc_aux_safety_loss(
                            batch["state"],
                            batch["action"],
                            disc_logits,
                            batch["labels_expert_is_one"].float(),
                        )

                reg_loss = self.safety_loss_weight * aux_loss
                loss = bce_loss + reg_loss

                assert len(batch["state"]) == 2 * self.demo_minibatch_size
                scale = self.demo_minibatch_size / self.demo_batch_size
                scaled_loss = loss * scale
                scaled_loss.backward()

                last_logits = disc_logits.detach()
                last_labels = batch["labels_expert_is_one"].detach()
                last_bce = bce_loss.detach()
                last_aux = aux_loss.detach()
                last_reg = reg_loss.detach()
                last_total = loss.detach()

                with th.no_grad():
                    labels = batch["labels_expert_is_one"].bool()
                    gen_labels = ~labels
                    if labels.any() and gen_labels.any():
                        last_reward_expert = raw_reward[labels].mean().detach()
                        last_reward_gen = raw_reward[gen_labels].mean().detach()
                        last_reward_gap = (last_reward_expert - last_reward_gen).detach()

            self._disc_opt.step()
            self._disc_step += 1

            with th.no_grad():
                train_stats = common.compute_train_stats(last_logits, last_labels, last_total)
                train_stats["disc_bce_loss"] = float(last_bce.item())
                train_stats["disc_safety_aux"] = float(last_aux.item())
                train_stats["disc_safety_reg_loss"] = float(last_reg.item())
                train_stats["disc_total_loss"] = float(last_total.item())
                train_stats["disc_safety_weight"] = float(self.safety_loss_weight)
                if last_reward_gap is not None:
                    train_stats["disc_reward_expert_mean"] = float(last_reward_expert.item())
                    train_stats["disc_reward_gen_mean"] = float(last_reward_gen.item())
                    train_stats["disc_reward_gap"] = float(last_reward_gap.item())

            self.logger.record("global_step", self._global_step)
            for k, v in train_stats.items():
                self.logger.record(k, v)
            self.logger.dump(self._disc_step)
        return train_stats
