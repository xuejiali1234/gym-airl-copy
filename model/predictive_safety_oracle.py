from __future__ import annotations

from typing import Optional

import torch


class PredictiveSafetyOracle:
    """Multi-step predictive safety oracle with soft risk labels."""

    def __init__(
        self,
        cfg,
        mean,
        std,
        *,
        horizon_steps: int = 10,
        dt: Optional[float] = None,
        collision_margin: float = 1.1,
        lane_x_tolerance: float = 10.0,
        ttc_lead_threshold: float = 3.0,
        ttc_follow_threshold: float = 3.0,
        thw_lead_threshold: float = 1.0,
        thw_follow_threshold: float = 0.8,
        gap_lead_threshold: float = 25.0,
        gap_follow_threshold: float = 25.0,
    ) -> None:
        self.cfg = cfg
        self.mean = torch.as_tensor(mean[:16], dtype=torch.float32)
        self.std = torch.as_tensor(std[:16], dtype=torch.float32)
        self.horizon_steps = int(horizon_steps)
        self.dt = float(dt if dt is not None else getattr(cfg, "DT", 0.1))
        self.collision_margin = float(collision_margin)
        self.lane_x_tolerance = float(lane_x_tolerance)
        self.ttc_lead_threshold = float(ttc_lead_threshold)
        self.ttc_follow_threshold = float(ttc_follow_threshold)
        self.thw_lead_threshold = float(thw_lead_threshold)
        self.thw_follow_threshold = float(thw_follow_threshold)
        self.gap_lead_threshold = float(gap_lead_threshold)
        self.gap_follow_threshold = float(gap_follow_threshold)

        self.vehicle_w = float(cfg.VEHICLE_WIDTH)
        self.vehicle_l = float(cfg.VEHICLE_LENGTH)
        self.phys_acc_max = float(cfg.PHYS_ACC_MAX)
        self.phys_steer_max = float(cfg.PHYS_STEER_MAX)
        self.speed_limit = float(getattr(cfg, "SPEED_LIMIT", 80.0))

    def _denormalize_state(self, states_tensor: torch.Tensor) -> torch.Tensor:
        core_state = states_tensor[..., :16]
        mean = self.mean.to(core_state.device)
        std = self.std.to(core_state.device)
        return core_state * std + mean

    def _denormalize_action(self, actions_tensor: torch.Tensor) -> torch.Tensor:
        actions = actions_tensor[..., :2]
        ax = actions[..., 0:1] * self.phys_steer_max
        ay = actions[..., 1:2] * self.phys_acc_max
        return torch.cat([ax, ay], dim=-1)

    def _rollout_one_step(self, raw_states: torch.Tensor, raw_actions: torch.Tensor) -> torch.Tensor:
        next_states = raw_states.clone()

        px, py, vx, vy = raw_states[:, 0], raw_states[:, 1], raw_states[:, 2], raw_states[:, 3]
        ax, ay = raw_actions[:, 0], raw_actions[:, 1]

        vx_new = vx + ax * self.dt
        vy_new = vy + ay * self.dt
        vy_new = torch.clamp(vy_new, min=0.0)

        speed = torch.sqrt(vx_new.pow(2) + vy_new.pow(2))
        overflow = speed > self.speed_limit
        if overflow.any():
            ratio = self.speed_limit / torch.clamp(speed, min=1e-6)
            vx_new = torch.where(overflow, vx_new * ratio, vx_new)
            vy_new = torch.where(overflow, vy_new * ratio, vy_new)

        px_new = px + vx_new * self.dt
        py_new = py + vy_new * self.dt

        next_states[:, 0] = px_new
        next_states[:, 1] = py_new
        next_states[:, 2] = vx_new
        next_states[:, 3] = vy_new

        for start in (4, 8, 12):
            dx = raw_states[:, start]
            dy = raw_states[:, start + 1]
            dvx = raw_states[:, start + 2]
            dvy = raw_states[:, start + 3]

            valid = (
                (torch.abs(dx) > 1e-5)
                | (torch.abs(dy) > 1e-5)
                | (torch.abs(dvx) > 1e-5)
                | (torch.abs(dvy) > 1e-5)
            )

            target_vx = vx + dvx
            target_vy = vy + dvy

            dx_next = dx + (target_vx - vx_new) * self.dt
            dy_next = dy + (target_vy - vy_new) * self.dt
            dvx_next = target_vx - vx_new
            dvy_next = target_vy - vy_new

            next_states[:, start] = torch.where(valid, dx_next, torch.zeros_like(dx_next))
            next_states[:, start + 1] = torch.where(valid, dy_next, torch.zeros_like(dy_next))
            next_states[:, start + 2] = torch.where(valid, dvx_next, torch.zeros_like(dvx_next))
            next_states[:, start + 3] = torch.where(valid, dvy_next, torch.zeros_like(dvy_next))

        return next_states

    def _slot_metrics(
        self,
        raw_states: torch.Tensor,
        start: int,
        *,
        is_follow: bool,
    ) -> dict[str, torch.Tensor]:
        dx = raw_states[:, start]
        dy = raw_states[:, start + 1]
        dvy = raw_states[:, start + 3]

        valid = (torch.abs(dx) > 1e-5) | (torch.abs(dy) > 1e-5)
        aligned = valid & (torch.abs(dx) <= self.lane_x_tolerance)

        if is_follow:
            longitudinal = -dy
            closing_speed = torch.clamp(dvy, min=0.0)
        else:
            longitudinal = dy
            closing_speed = torch.clamp(-dvy, min=0.0)

        positive_gap = aligned & (longitudinal > 0.1)

        huge = torch.full_like(longitudinal, 1e6)
        ttc = torch.where(
            positive_gap & (closing_speed > 0.1),
            longitudinal / torch.clamp(closing_speed, min=1e-3),
            huge,
        )

        ego_vy = raw_states[:, 3]
        thw = torch.where(
            positive_gap & (ego_vy > 0.1),
            longitudinal / torch.clamp(ego_vy, min=1e-3),
            huge,
        )
        gap = torch.where(positive_gap, longitudinal, huge)
        overlap = aligned & (torch.abs(dx) < self.vehicle_w) & (torch.abs(dy) < self.vehicle_l)
        return {
            "ttc": ttc,
            "thw": thw,
            "gap": gap,
            "overlap": overlap,
        }

    @staticmethod
    def _risk_from_threshold(values: torch.Tensor, threshold: float) -> torch.Tensor:
        threshold = float(threshold)
        return torch.clamp((threshold - values) / max(threshold, 1e-6), min=0.0, max=1.0)

    def analyze_batch(self, states_tensor: torch.Tensor, actions: Optional[torch.Tensor] = None) -> dict[str, torch.Tensor]:
        raw_states = self._denormalize_state(states_tensor)
        if actions is None:
            raw_actions = raw_states.new_zeros(raw_states.shape[0], 2)
        else:
            raw_actions = self._denormalize_action(actions)

        device = raw_states.device
        batch_size = raw_states.shape[0]
        inf = raw_states.new_full((batch_size,), 1e6)

        min_ttc_lead = inf.clone()
        min_ttc_follow = inf.clone()
        min_thw_lead = inf.clone()
        min_thw_follow = inf.clone()
        min_gap_lead = inf.clone()
        min_gap_follow = inf.clone()
        future_overlap = torch.zeros(batch_size, dtype=torch.bool, device=device)

        rollout_states = raw_states
        for _ in range(self.horizon_steps):
            lead_l6 = self._slot_metrics(rollout_states, 4, is_follow=False)
            lead_l5 = self._slot_metrics(rollout_states, 8, is_follow=False)
            follow_l5 = self._slot_metrics(rollout_states, 12, is_follow=True)

            min_ttc_lead = torch.minimum(min_ttc_lead, torch.minimum(lead_l6["ttc"], lead_l5["ttc"]))
            min_ttc_follow = torch.minimum(min_ttc_follow, follow_l5["ttc"])
            min_thw_lead = torch.minimum(min_thw_lead, torch.minimum(lead_l6["thw"], lead_l5["thw"]))
            min_thw_follow = torch.minimum(min_thw_follow, follow_l5["thw"])
            min_gap_lead = torch.minimum(min_gap_lead, torch.minimum(lead_l6["gap"], lead_l5["gap"]))
            min_gap_follow = torch.minimum(min_gap_follow, follow_l5["gap"])
            future_overlap = future_overlap | lead_l6["overlap"] | lead_l5["overlap"] | follow_l5["overlap"]

            rollout_states = self._rollout_one_step(rollout_states, raw_actions)

        overlap_risk = future_overlap.float()
        ttc_lead_risk = self._risk_from_threshold(min_ttc_lead, self.ttc_lead_threshold)
        ttc_follow_risk = self._risk_from_threshold(min_ttc_follow, self.ttc_follow_threshold)
        thw_lead_risk = self._risk_from_threshold(min_thw_lead, self.thw_lead_threshold)
        thw_follow_risk = self._risk_from_threshold(min_thw_follow, self.thw_follow_threshold)
        gap_lead_risk = self._risk_from_threshold(min_gap_lead, self.gap_lead_threshold)
        gap_follow_risk = self._risk_from_threshold(min_gap_follow, self.gap_follow_threshold)

        risk_score = torch.clamp(
            overlap_risk
            + 0.5 * torch.maximum(ttc_lead_risk, ttc_follow_risk)
            + 0.5 * torch.maximum(thw_lead_risk, thw_follow_risk)
            + 0.3 * torch.maximum(gap_lead_risk, gap_follow_risk),
            min=0.0,
            max=1.0,
        )

        critical = (
            future_overlap
            | (min_ttc_lead < 1.0)
            | (min_ttc_follow < 1.0)
            | (min_thw_lead < 0.25)
            | (min_thw_follow < 0.25)
        )

        return {
            "risk_score": risk_score.unsqueeze(1),
            "critical_label": critical.float().unsqueeze(1),
            "min_ttc_lead": min_ttc_lead.unsqueeze(1),
            "min_ttc_follow": min_ttc_follow.unsqueeze(1),
            "min_thw_lead": min_thw_lead.unsqueeze(1),
            "min_thw_follow": min_thw_follow.unsqueeze(1),
            "min_gap_lead": min_gap_lead.unsqueeze(1),
            "min_gap_follow": min_gap_follow.unsqueeze(1),
            "future_overlap": future_overlap.float().unsqueeze(1),
        }

    def get_labels(self, states_tensor: torch.Tensor, actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.analyze_batch(states_tensor, actions)["risk_score"]

    def get_critical_labels(self, states_tensor: torch.Tensor, actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.analyze_batch(states_tensor, actions)["critical_label"]

    def check_safety_batch(self, states_tensor: torch.Tensor, actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.get_labels(states_tensor, actions)
