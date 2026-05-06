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
        safety_reg_mode: str = "legacy_aux",
        safety_reg_margin: float = 0.2,
        safety_cpair_additive_weight: float = 0.0,
        safety_candidate_set: str = "current",
        safety_safe_selection: str = "min_risk",
        safety_rank_metric: str = "clipped",
        safety_late_tiny_cpair_weight: float = 0.0,
        safety_late_tiny_candidate_set: str = "candidate_v2_leadaware",
        safety_late_tiny_safe_selection: str = "task_safe",
        safety_late_tiny_rank_metric: str = "clipped",
        safety_late_tiny_reg_margin: float = 0.1,
        safety_focused_cpair_enable: bool = False,
        safety_focused_cpair_min_gap: float = 0.10,
        safety_focused_cpair_min_policy_risk: float = 0.30,
        safety_focused_cpair_weight_clip: float = 0.20,
        safety_focused_cpair_weight_source: str = "raw",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.safety_loss_weight = float(safety_loss_weight)
        self.safety_reg_mode = str(safety_reg_mode)
        self.safety_reg_margin = float(safety_reg_margin)
        self.safety_cpair_additive_weight = float(safety_cpair_additive_weight)
        self.safety_candidate_set = str(safety_candidate_set)
        self.safety_safe_selection = str(safety_safe_selection)
        self.safety_rank_metric = str(safety_rank_metric)
        self.safety_late_tiny_cpair_weight = float(safety_late_tiny_cpair_weight)
        self.safety_late_tiny_candidate_set = str(safety_late_tiny_candidate_set)
        self.safety_late_tiny_safe_selection = str(safety_late_tiny_safe_selection)
        self.safety_late_tiny_rank_metric = str(safety_late_tiny_rank_metric)
        self.safety_late_tiny_reg_margin = float(safety_late_tiny_reg_margin)
        self.safety_focused_cpair_enable = bool(safety_focused_cpair_enable)
        self.safety_focused_cpair_min_gap = float(safety_focused_cpair_min_gap)
        self.safety_focused_cpair_min_policy_risk = float(safety_focused_cpair_min_policy_risk)
        self.safety_focused_cpair_weight_clip = float(safety_focused_cpair_weight_clip)
        self.safety_focused_cpair_weight_source = str(safety_focused_cpair_weight_source)

    def set_safety_aux_weights(
        self,
        *,
        legacy_weight: Optional[float] = None,
        candidate_pair_additive_weight: Optional[float] = None,
        late_tiny_candidate_pair_weight: Optional[float] = None,
    ) -> None:
        if legacy_weight is not None:
            self.safety_loss_weight = float(legacy_weight)
        if candidate_pair_additive_weight is not None:
            self.safety_cpair_additive_weight = float(candidate_pair_additive_weight)
        if late_tiny_candidate_pair_weight is not None:
            self.safety_late_tiny_cpair_weight = float(late_tiny_candidate_pair_weight)

    def _build_predictive_candidate_actions(
        self,
        action: th.Tensor,
        candidate_set: Optional[str] = None,
    ) -> tuple[th.Tensor, list[str]]:
        """Construct same-state candidate actions for predictive safety ranking."""
        lateral = action[:, 0]
        longitudinal = action[:, 1]

        hold_lane = th.stack([th.zeros_like(lateral), longitudinal], dim=-1)
        delay_merge = th.stack([0.5 * lateral.clamp(max=0.0), longitudinal], dim=-1)
        mild_decel = th.stack([lateral, (longitudinal - 0.25).clamp(-1.0, 1.0)], dim=-1)
        mild_accel = th.stack([lateral, (longitudinal + 0.25).clamp(-1.0, 1.0)], dim=-1)
        mild_merge = th.stack([(lateral - 0.25).clamp(-1.0, 1.0), longitudinal], dim=-1)

        hold_decel = th.stack([th.zeros_like(lateral), (longitudinal - 0.25).clamp(-1.0, 1.0)], dim=-1)
        strong_decel = th.stack([lateral, (longitudinal - 0.5).clamp(-1.0, 1.0)], dim=-1)
        delay_decel = th.stack([0.5 * lateral.clamp(max=0.0), (longitudinal - 0.25).clamp(-1.0, 1.0)], dim=-1)
        speedmatch = th.stack([lateral, (longitudinal - 0.12).clamp(-1.0, 1.0)], dim=-1)
        delay_speedmatch = th.stack([0.5 * lateral.clamp(max=0.0), (longitudinal - 0.12).clamp(-1.0, 1.0)], dim=-1)
        merge_decel = th.stack([(lateral - 0.25).clamp(-1.0, 1.0), (longitudinal - 0.25).clamp(-1.0, 1.0)], dim=-1)
        merge_speedmatch = th.stack([(lateral - 0.25).clamp(-1.0, 1.0), (longitudinal - 0.12).clamp(-1.0, 1.0)], dim=-1)
        strong_merge = th.stack([(lateral - 0.5).clamp(-1.0, 1.0), longitudinal], dim=-1)
        strong_merge_decel = th.stack([(lateral - 0.5).clamp(-1.0, 1.0), (longitudinal - 0.25).clamp(-1.0, 1.0)], dim=-1)
        strong_merge_accel = th.stack([(lateral - 0.5).clamp(-1.0, 1.0), (longitudinal + 0.25).clamp(-1.0, 1.0)], dim=-1)

        candidate_groups = {
            "current": [
                ("policy", action),
                ("hold", hold_lane),
                ("delay", delay_merge),
                ("decel", mild_decel),
                ("accel", mild_accel),
                ("merge", mild_merge),
            ],
            "current_mini_lead": [
                ("policy", action),
                ("hold", hold_lane),
                ("delay", delay_merge),
                ("decel", mild_decel),
                ("accel", mild_accel),
                ("merge", mild_merge),
                ("delay_decel", delay_decel),
                ("speedmatch", speedmatch),
                ("merge_decel", merge_decel),
            ],
            "candidate_v2": [
                ("policy", action),
                ("hold", hold_lane),
                ("delay", delay_merge),
                ("decel", mild_decel),
                ("accel", mild_accel),
                ("merge", mild_merge),
                ("hold_decel", hold_decel),
                ("delay_decel", delay_decel),
                ("speedmatch", speedmatch),
                ("merge_decel", merge_decel),
            ],
            "candidate_v2_leadaware": [
                ("policy", action),
                ("hold", hold_lane),
                ("delay", delay_merge),
                ("decel", mild_decel),
                ("accel", mild_accel),
                ("merge", mild_merge),
                ("hold_decel", hold_decel),
                ("delay_decel", delay_decel),
                ("speedmatch", speedmatch),
                ("merge_decel", merge_decel),
                ("strong_decel", strong_decel),
                ("delay_speedmatch", delay_speedmatch),
                ("merge_speedmatch", merge_speedmatch),
                ("strong_merge", strong_merge),
                ("strong_merge_decel", strong_merge_decel),
                ("strong_merge_accel", strong_merge_accel),
            ],
        }
        selected = candidate_groups.get(candidate_set or self.safety_candidate_set, candidate_groups["current"])
        names = [name for name, _ in selected]
        tensors = [tensor for _, tensor in selected]
        candidates = th.stack(tensors, dim=1).clamp(-1.0, 1.0)
        return candidates, names

    @staticmethod
    def _action_category(name: str) -> str:
        if name == "policy":
            return "policy"
        if "speedmatch" in name:
            return "speedmatch"
        if "delay" in name:
            return "delay"
        if "merge" in name:
            return "merge"
        if "decel" in name:
            return "decel"
        if "hold" in name:
            return "hold"
        if "accel" in name:
            return "accel"
        return "other"

    def _task_safe_score(self, candidate_risk: th.Tensor, candidate_names: list[str]) -> th.Tensor:
        score = candidate_risk.clone()
        for idx, name in enumerate(candidate_names):
            category = self._action_category(name)
            if category == "merge":
                score[:, idx] -= 0.035
            elif category == "delay":
                score[:, idx] -= 0.025
            elif category == "speedmatch":
                score[:, idx] -= 0.015
            elif category == "policy":
                score[:, idx] -= 0.015
            elif category == "decel":
                score[:, idx] -= 0.010

            if name == "hold":
                score[:, idx] += 0.020
            if name == "strong_decel":
                score[:, idx] += 0.040
            if name == "strong_merge_accel":
                score[:, idx] += 0.030
            if name == "accel":
                score[:, idx] += 0.015
        return score

    def _predictive_candidate_pair_loss(
        self,
        *,
        base_reward_net,
        batch,
        candidate_set: Optional[str] = None,
        safe_selection: Optional[str] = None,
        rank_metric: Optional[str] = None,
        reg_margin: Optional[float] = None,
    ) -> tuple[th.Tensor, dict[str, th.Tensor]]:
        candidate_set = str(candidate_set or self.safety_candidate_set)
        safe_selection = str(safe_selection or self.safety_safe_selection)
        rank_metric = str(rank_metric or self.safety_rank_metric)
        reg_margin = float(self.safety_reg_margin if reg_margin is None else reg_margin)

        def _zero_stats(device: th.device) -> dict[str, th.Tensor]:
            zero = th.zeros((), device=device)
            return {
                "cpair_valid_rate": zero,
                "cpair_raw_risk_gap_mean": zero,
                "cpair_clipped_risk_gap_mean": zero,
                "cpair_all_candidates_saturated_rate": zero,
                "cpair_policy_risk_raw_mean": zero,
                "cpair_safe_risk_raw_mean": zero,
                "cpair_risky_risk_raw_mean": zero,
                "cpair_safe_type_hold_rate": zero,
                "cpair_safe_type_decel_rate": zero,
                "cpair_safe_type_delay_rate": zero,
                "cpair_safe_type_merge_rate": zero,
                "cpair_safe_type_speedmatch_rate": zero,
                "cpair_risky_type_policy_rate": zero,
                "cpair_reward_gap_mean": zero,
                "cpair_focused_active_rate": zero,
            }

        labels = batch["labels_expert_is_one"] < 0.5
        if not labels.any():
            return th.zeros((), device=batch["state"].device), _zero_stats(batch["state"].device)

        state = batch["state"][labels]
        action = batch["action"][labels]
        next_state = batch["next_state"][labels]
        done = batch["done"][labels]
        num_samples = state.shape[0]

        candidate_actions, candidate_names = self._build_predictive_candidate_actions(
            action,
            candidate_set=candidate_set,
        )
        num_candidates = candidate_actions.shape[1]

        state_rep = state.unsqueeze(1).expand(-1, num_candidates, -1).reshape(-1, state.shape[-1])
        next_state_rep = next_state.unsqueeze(1).expand(-1, num_candidates, -1).reshape(-1, next_state.shape[-1])
        done_rep = done.unsqueeze(1).expand(-1, num_candidates).reshape(-1)
        candidate_actions_flat = candidate_actions.reshape(-1, action.shape[-1])

        safety_debug = base_reward_net.get_safety_debug(state_rep, candidate_actions_flat)
        candidate_risk_clipped = safety_debug["q_safe_risk"].reshape(num_samples, num_candidates)
        candidate_risk_raw = safety_debug["q_safe_logit"].reshape(num_samples, num_candidates)
        if rank_metric == "raw":
            candidate_risk = candidate_risk_raw
        else:
            candidate_risk = candidate_risk_clipped

        candidate_reward = self._reward_net(
            state_rep,
            candidate_actions_flat,
            next_state_rep,
            done_rep,
        ).reshape(num_samples, num_candidates)

        if safe_selection == "task_safe":
            task_scores = self._task_safe_score(candidate_risk, candidate_names)
            policy_risk = candidate_risk[:, 0]
            eligible_mask = candidate_risk < (policy_risk.unsqueeze(1) - 1e-4)
            eligible_mask[:, 0] = False
            masked_scores = task_scores.masked_fill(~eligible_mask, float("inf"))
            has_task_safe = eligible_mask.any(dim=1)
            min_task_idx = th.argmin(masked_scores, dim=1)
            min_risk_idx = th.argmin(candidate_risk, dim=1)
            safe_idx = th.where(has_task_safe, min_task_idx, min_risk_idx)
        else:
            safe_idx = th.argmin(candidate_risk, dim=1)
        max_risk_idx = th.argmax(candidate_risk, dim=1)
        policy_risk = candidate_risk[:, 0]
        policy_risk_clipped = candidate_risk_clipped[:, 0]
        risky_idx = th.where(policy_risk_clipped >= 0.7, th.zeros_like(max_risk_idx), max_risk_idx)

        row_idx = th.arange(num_samples, device=state.device)
        safe_reward = candidate_reward[row_idx, safe_idx]
        risky_reward = candidate_reward[row_idx, risky_idx]
        safe_risk = candidate_risk[row_idx, safe_idx]
        risky_risk = candidate_risk[row_idx, risky_idx]
        safe_risk_raw = candidate_risk_raw[row_idx, safe_idx]
        risky_risk_raw = candidate_risk_raw[row_idx, risky_idx]
        safe_risk_clipped = candidate_risk_clipped[row_idx, safe_idx]
        risky_risk_clipped = candidate_risk_clipped[row_idx, risky_idx]

        risk_gap = (risky_risk - safe_risk).detach()
        raw_risk_gap = (risky_risk_raw - safe_risk_raw).detach()
        clipped_risk_gap = (risky_risk_clipped - safe_risk_clipped).detach()
        valid_mask = risk_gap > 1e-4
        all_candidates_saturated = (candidate_risk_clipped >= 0.999).all(dim=1)

        safe_type_rates: dict[str, th.Tensor] = {}
        category_names = ["hold", "decel", "delay", "merge", "speedmatch"]
        safe_type_categories = [self._action_category(name) for name in candidate_names]
        for category in category_names:
            match_count = sum(1 for item in safe_type_categories if item == category)
            if match_count == 0:
                safe_type_rates[category] = th.zeros((), device=state.device)
                continue
            indicator = state.new_tensor(
                [1.0 if item == category else 0.0 for item in safe_type_categories],
            )
            safe_type_rates[category] = indicator[safe_idx].mean()

        risky_type_policy_rate = (risky_idx == 0).float().mean()
        reward_gap = (safe_reward - risky_reward).detach()

        stats = _zero_stats(state.device)
        stats["cpair_valid_rate"] = valid_mask.float().mean()
        stats["cpair_raw_risk_gap_mean"] = raw_risk_gap.mean()
        stats["cpair_clipped_risk_gap_mean"] = clipped_risk_gap.mean()
        stats["cpair_all_candidates_saturated_rate"] = all_candidates_saturated.float().mean()
        stats["cpair_policy_risk_raw_mean"] = candidate_risk_raw[:, 0].mean()
        stats["cpair_safe_risk_raw_mean"] = safe_risk_raw.mean()
        stats["cpair_risky_risk_raw_mean"] = risky_risk_raw.mean()
        stats["cpair_safe_type_hold_rate"] = safe_type_rates["hold"]
        stats["cpair_safe_type_decel_rate"] = safe_type_rates["decel"]
        stats["cpair_safe_type_delay_rate"] = safe_type_rates["delay"]
        stats["cpair_safe_type_merge_rate"] = safe_type_rates["merge"]
        stats["cpair_safe_type_speedmatch_rate"] = safe_type_rates["speedmatch"]
        stats["cpair_risky_type_policy_rate"] = risky_type_policy_rate
        stats["cpair_reward_gap_mean"] = reward_gap.mean()

        pair_loss = F.relu(reg_margin + risky_reward - safe_reward)
        if self.safety_focused_cpair_enable:
            focused_mask = (
                valid_mask
                & (clipped_risk_gap >= self.safety_focused_cpair_min_gap)
                & (policy_risk_clipped >= self.safety_focused_cpair_min_policy_risk)
            )
            stats["cpair_focused_active_rate"] = focused_mask.float().mean()
            if not focused_mask.any():
                return th.zeros((), device=state.device), stats
            if self.safety_focused_cpair_weight_source == "clipped":
                focused_weight = clipped_risk_gap.clamp(min=0.0, max=self.safety_focused_cpair_weight_clip)
            else:
                focused_weight = risk_gap.clamp(min=0.0, max=self.safety_focused_cpair_weight_clip)
            weighted_loss = pair_loss[focused_mask] * focused_weight[focused_mask]
            return weighted_loss.mean(), stats

        if not valid_mask.any():
            return th.zeros((), device=state.device), stats
        weighted_loss = pair_loss[valid_mask] * risk_gap[valid_mask]
        return weighted_loss.mean(), stats

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
            last_cpair_aux = None
            last_cpair_reg = None
            last_cpair_stats = None
            last_late_tiny_cpair_aux = None
            last_late_tiny_cpair_reg = None
            last_late_tiny_cpair_stats = None
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
                cpair_aux_loss = th.zeros((), device=disc_logits.device)
                late_tiny_cpair_aux_loss = th.zeros((), device=disc_logits.device)
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
                        aux_loss, cpair_stats = self._predictive_candidate_pair_loss(
                            base_reward_net=base_reward_net,
                            batch=batch,
                        )
                        last_cpair_stats = cpair_stats
                    elif hasattr(base_reward_net, "disc_aux_safety_loss"):
                        aux_loss = base_reward_net.disc_aux_safety_loss(
                            batch["state"],
                            batch["action"],
                            disc_logits,
                            batch["labels_expert_is_one"].float(),
                        )

                if (
                    self.safety_cpair_additive_weight > 0.0
                    and self.safety_reg_mode != "predictive_candidate_ranking"
                    and hasattr(base_reward_net, "get_safety_debug")
                ):
                    cpair_aux_loss, cpair_stats = self._predictive_candidate_pair_loss(
                        base_reward_net=base_reward_net,
                        batch=batch,
                    )
                    last_cpair_stats = cpair_stats

                if (
                    self.safety_late_tiny_cpair_weight > 0.0
                    and hasattr(base_reward_net, "get_safety_debug")
                ):
                    late_tiny_cpair_aux_loss, late_tiny_cpair_stats = self._predictive_candidate_pair_loss(
                        base_reward_net=base_reward_net,
                        batch=batch,
                        candidate_set=self.safety_late_tiny_candidate_set,
                        safe_selection=self.safety_late_tiny_safe_selection,
                        rank_metric=self.safety_late_tiny_rank_metric,
                        reg_margin=self.safety_late_tiny_reg_margin,
                    )
                    last_late_tiny_cpair_stats = late_tiny_cpair_stats

                reg_loss = (
                    self.safety_loss_weight * aux_loss
                    + self.safety_cpair_additive_weight * cpair_aux_loss
                    + self.safety_late_tiny_cpair_weight * late_tiny_cpair_aux_loss
                )
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
                last_cpair_aux = cpair_aux_loss.detach()
                last_cpair_reg = (self.safety_cpair_additive_weight * cpair_aux_loss).detach()
                last_late_tiny_cpair_aux = late_tiny_cpair_aux_loss.detach()
                last_late_tiny_cpair_reg = (
                    self.safety_late_tiny_cpair_weight * late_tiny_cpair_aux_loss
                ).detach()
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
                train_stats["disc_cpair_aux"] = float(last_cpair_aux.item())
                train_stats["disc_cpair_reg_loss"] = float(last_cpair_reg.item())
                train_stats["disc_cpair_late_tiny_aux"] = float(last_late_tiny_cpair_aux.item())
                train_stats["disc_cpair_late_tiny_reg_loss"] = float(last_late_tiny_cpair_reg.item())
                train_stats["disc_total_loss"] = float(last_total.item())
                train_stats["disc_safety_weight"] = float(self.safety_loss_weight)
                train_stats["disc_safety_cpair_additive_weight"] = float(self.safety_cpair_additive_weight)
                train_stats["disc_safety_late_tiny_cpair_weight"] = float(self.safety_late_tiny_cpair_weight)
                if last_cpair_stats is not None:
                    for key, value in last_cpair_stats.items():
                        train_stats[key] = float(value.item())
                if last_late_tiny_cpair_stats is not None:
                    for key, value in last_late_tiny_cpair_stats.items():
                        train_stats[f"late_tiny_{key}"] = float(value.item())
                if last_reward_gap is not None:
                    train_stats["disc_reward_expert_mean"] = float(last_reward_expert.item())
                    train_stats["disc_reward_gen_mean"] = float(last_reward_gen.item())
                    train_stats["disc_reward_gap"] = float(last_reward_gap.item())

            self.logger.record("global_step", self._global_step)
            for k, v in train_stats.items():
                self.logger.record(k, v)
            self.logger.dump(self._disc_step)
        return train_stats
