from __future__ import annotations

from typing import Optional

import torch


class SafetyOracleQ:
    """
    State-action safety oracle.

    It labels (s, a) as unsafe if either:
      1) the current state is already unsafe, or
      2) a one-step rollout under action a enters a collision / low-TTC region.

    The rollout uses the same 0.1s kinematic convention as your Gym environment,
    while surrounding vehicles are propagated with constant velocity inferred from
    the relative-state representation.
    """

    def __init__(
        self,
        cfg,
        mean,
        std,
        *,
        collision_margin: float = 1.1,
        ttc_threshold: float = 3.0,
        warning_ttc_threshold: Optional[float] = None,
        warning_weight: float = 1.0,
        lane_x_tolerance: float = 10.0,
    ) -> None:
        self.cfg = cfg
        self.mean = torch.as_tensor(mean[:16], dtype=torch.float32)
        self.std = torch.as_tensor(std[:16], dtype=torch.float32)
        self.collision_margin = float(collision_margin)
        self.ttc_threshold = float(getattr(cfg, "SAFETY_ORACLE_TTC_THRESHOLD", ttc_threshold))
        cfg_warning = getattr(cfg, "SAFETY_ORACLE_WARNING_TTC_THRESHOLD", warning_ttc_threshold)
        self.warning_ttc_threshold = None if cfg_warning is None else float(cfg_warning)
        self.warning_weight = float(getattr(cfg, "SAFETY_ORACLE_WARNING_WEIGHT", warning_weight))
        self.lane_x_tolerance = float(lane_x_tolerance)

        self.vehicle_w = float(cfg.VEHICLE_WIDTH)
        self.vehicle_l = float(cfg.VEHICLE_LENGTH)
        self.dt = float(cfg.DT)
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

    def _is_collision(self, dx: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
        safe_dx = (self.vehicle_w * self.collision_margin + self.vehicle_w) / 2.0
        safe_dy = (self.vehicle_l * self.collision_margin + self.vehicle_l) / 2.0
        is_valid_car = (torch.abs(dx) > 0.01) | (torch.abs(dy) > 0.01)
        is_overlapping = (torch.abs(dx) < safe_dx) & (torch.abs(dy) < safe_dy)
        return is_valid_car & is_overlapping

    def _is_low_ttc(self, dx: torch.Tensor, dy: torch.Tensor, dvy: torch.Tensor) -> torch.Tensor:
        valid_x = torch.abs(dx) <= self.lane_x_tolerance
        valid_y = dy > 0.1
        rel_speed = -dvy
        valid_speed = rel_speed > 0.1
        safe_rel_speed = torch.clamp(rel_speed, min=1e-3)
        ttc = dy / safe_rel_speed
        valid_ttc = ttc < self.ttc_threshold
        return valid_x & valid_y & valid_speed & valid_ttc

    def _ttc_warning_risk(self, dx: torch.Tensor, dy: torch.Tensor, dvy: torch.Tensor) -> torch.Tensor:
        if self.warning_ttc_threshold is None or self.warning_ttc_threshold <= self.ttc_threshold:
            return torch.zeros_like(dy)

        valid_x = torch.abs(dx) <= self.lane_x_tolerance
        valid_y = dy > 0.1
        rel_speed = -dvy
        valid_speed = rel_speed > 0.1
        safe_rel_speed = torch.clamp(rel_speed, min=1e-3)
        ttc = dy / safe_rel_speed

        warning = torch.clamp((self.warning_ttc_threshold - ttc) / self.warning_ttc_threshold, min=0.0, max=1.0)
        warning = warning * float(self.warning_weight)
        active = valid_x & valid_y & valid_speed & (ttc >= self.ttc_threshold) & (ttc < self.warning_ttc_threshold)
        return torch.where(active, warning, torch.zeros_like(warning))

    def _unsafe_from_raw_state(self, raw_states: torch.Tensor) -> torch.Tensor:
        dx_l6, dy_l6, _, dvy_l6 = raw_states[:, 4], raw_states[:, 5], raw_states[:, 6], raw_states[:, 7]
        dx_l5, dy_l5, _, dvy_l5 = raw_states[:, 8], raw_states[:, 9], raw_states[:, 10], raw_states[:, 11]
        dx_f, dy_f = raw_states[:, 12], raw_states[:, 13]

        collision_risk = (
            self._is_collision(dx_l6, dy_l6)
            | self._is_collision(dx_l5, dy_l5)
            | self._is_collision(dx_f, dy_f)
        )
        ttc_risk = self._is_low_ttc(dx_l6, dy_l6, dvy_l6) | self._is_low_ttc(dx_l5, dy_l5, dvy_l5)
        is_unsafe = collision_risk | ttc_risk
        return is_unsafe.float().unsqueeze(1)

    def _soft_risk_from_raw_state(self, raw_states: torch.Tensor) -> torch.Tensor:
        hard_risk = self._unsafe_from_raw_state(raw_states)
        dx_l6, dy_l6, _, dvy_l6 = raw_states[:, 4], raw_states[:, 5], raw_states[:, 6], raw_states[:, 7]
        dx_l5, dy_l5, _, dvy_l5 = raw_states[:, 8], raw_states[:, 9], raw_states[:, 10], raw_states[:, 11]
        warning = torch.maximum(
            self._ttc_warning_risk(dx_l6, dy_l6, dvy_l6),
            self._ttc_warning_risk(dx_l5, dy_l5, dvy_l5),
        ).unsqueeze(1)
        return torch.maximum(hard_risk, warning).clamp(0.0, 1.0)

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

            # Missing vehicle slots are all zeros in your normalized/raw representation.
            valid = (torch.abs(dx) > 1e-5) | (torch.abs(dy) > 1e-5) | (torch.abs(dvx) > 1e-5) | (torch.abs(dvy) > 1e-5)

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

    def get_labels(self, states_tensor: torch.Tensor, actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        raw_states = self._denormalize_state(states_tensor)
        current_risk = self._soft_risk_from_raw_state(raw_states)
        if actions is None:
            return current_risk

        raw_actions = self._denormalize_action(actions)
        raw_next = self._rollout_one_step(raw_states, raw_actions)
        next_risk = self._soft_risk_from_raw_state(raw_next)
        return torch.maximum(current_risk, next_risk)

    def check_safety_batch(self, states_tensor: torch.Tensor, actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.get_labels(states_tensor, actions)
