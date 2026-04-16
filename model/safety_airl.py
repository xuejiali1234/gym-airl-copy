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

    def __init__(self, *args, safety_loss_weight: float = 0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.safety_loss_weight = float(safety_loss_weight)

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

                aux_loss = th.zeros((), device=disc_logits.device)
                if self.safety_loss_weight > 0.0 and hasattr(base_reward_net, "disc_aux_safety_loss"):
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

            self._disc_opt.step()
            self._disc_step += 1

            with th.no_grad():
                train_stats = common.compute_train_stats(last_logits, last_labels, last_total)
                train_stats["disc_bce_loss"] = float(last_bce.item())
                train_stats["disc_safety_aux"] = float(last_aux.item())
                train_stats["disc_safety_reg_loss"] = float(last_reg.item())
                train_stats["disc_total_loss"] = float(last_total.item())
                train_stats["disc_safety_weight"] = float(self.safety_loss_weight)

            self.logger.record("global_step", self._global_step)
            for k, v in train_stats.items():
                self.logger.record(k, v)
            self.logger.dump(self._disc_step)
        return train_stats
