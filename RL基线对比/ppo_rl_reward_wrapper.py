from collections import defaultdict

import gymnasium as gym
import numpy as np


class PpoRlRewardWrapper(gym.Wrapper):
    """Replace the sparse env reward with a handcrafted PPO-RL reward."""

    def __init__(self, env, cfg):
        super().__init__(env)
        self.cfg = cfg
        self.speed_limit_mps = float(getattr(cfg, "SPEED_LIMIT", 80.0)) * 0.3048
        self.eff_mode = str(getattr(cfg, "PPO_RL_EFF_MODE", "speed_limit")).strip().lower()
        self.target_speed_mps = float(getattr(cfg, "PPO_RL_TARGET_SPEED_MPS", 15.0))
        self.target_speed_band_mps = float(getattr(cfg, "PPO_RL_TARGET_SPEED_BAND_MPS", 5.0))
        self.speed_over_mps = float(getattr(cfg, "PPO_RL_SPEED_OVER_MPS", 17.0))
        self.speed_over_band_mps = float(getattr(cfg, "PPO_RL_SPEED_OVER_BAND_MPS", 5.0))
        if self.eff_mode not in {"speed_limit", "target_speed"}:
            raise ValueError(
                "PPO_RL_EFF_MODE must be 'speed_limit' or 'target_speed', "
                f"got {self.eff_mode!r}"
            )
        self.weights = {
            "eff": float(getattr(cfg, "PPO_RL_W_EFF", 0.20)),
            "safety": float(getattr(cfg, "PPO_RL_W_SAFETY", 1.00)),
            "thw": float(getattr(cfg, "PPO_RL_W_THW", 0.30)),
            "comfort": float(getattr(cfg, "PPO_RL_W_COMFORT", 0.05)),
            "goal": float(getattr(cfg, "PPO_RL_W_GOAL", 0.80)),
            "speed_over": float(getattr(cfg, "PPO_RL_W_SPEED_OVER", 0.0)),
        }
        self.collision_penalty = float(getattr(cfg, "PPO_RL_COLLISION_PENALTY", -5.0))
        self.success_bonus = float(getattr(cfg, "PPO_RL_SUCCESS_BONUS", 1.0))
        self.merge_bonus = float(getattr(cfg, "PPO_RL_MERGE_BONUS", 0.5))
        self.timeout_penalty = float(getattr(cfg, "PPO_RL_TIMEOUT_PENALTY", -1.0))
        self.reward_clip_min = float(getattr(cfg, "PPO_RL_REWARD_CLIP_MIN", -10.0))
        self.reward_clip_max = float(getattr(cfg, "PPO_RL_REWARD_CLIP_MAX", 3.0))
        self.jerk_norm = float(getattr(cfg, "PPO_RL_JERK_NORM", 3.0))
        self.jerk_x_norm = float(getattr(cfg, "PPO_RL_JERK_X_NORM", self.jerk_norm))
        self.jerk_y_norm = float(getattr(cfg, "PPO_RL_JERK_Y_NORM", self.jerk_norm))
        self.thw_safe_seconds = float(getattr(cfg, "PPO_RL_THW_SAFE_SECONDS", 2.0))
        self.completed_episode_summaries = []
        self._reset_episode_trackers()

    def _reset_episode_trackers(self):
        self._episode_steps = 0
        self._episode_return = 0.0
        self._episode_term_sums = defaultdict(float)
        self._episode_event_counts = defaultdict(float)
        self._has_merged = False
        self._has_collided = False

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._reset_episode_trackers()
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        terms = self._compute_reward_terms(info, terminated, truncated)
        reward = self._combine_terms(terms)

        info["ppo_rl_reward_terms"] = dict(terms)
        info["ppo_rl_reward"] = float(reward)

        self._episode_steps += 1
        self._episode_return += float(reward)
        for key, value in terms.items():
            self._episode_term_sums[key] += float(value)
        for key in ("merge_event", "success_event", "timeout_event", "collision_event"):
            self._episode_event_counts[key] += float(terms.get(key, 0.0))

        if terminated or truncated:
            self._finalize_episode()

        return obs, float(reward), terminated, truncated, info

    def _safe_longitudinal_speed_mps(self):
        ego_state = getattr(self.unwrapped, "ego_state", None)
        if ego_state is None or len(ego_state) < 4:
            return 0.0
        return float(max(ego_state[3], 0.0) * 0.3048)

    def _compute_eff_score(self, vy_mps):
        if self.eff_mode == "target_speed":
            norm_error = (vy_mps - self.target_speed_mps) / max(self.target_speed_band_mps, 1e-6)
            return float(np.clip(1.0 - norm_error ** 2, -1.0, 1.0))
        return float(np.clip(vy_mps / max(self.speed_limit_mps, 1e-6), 0.0, 1.0))

    def _compute_reward_terms(self, info, terminated, truncated):
        is_collided = bool(info.get("is_collided", False))
        just_collided = is_collided and (not self._has_collided)
        if just_collided:
            self._has_collided = True

        is_merge_success = bool(info.get("is_merge_success", False))
        just_merged = is_merge_success and (not self._has_merged)
        if just_merged:
            self._has_merged = True

        is_endpoint_success = bool(info.get("is_endpoint_success", False))

        vy_mps = float(info.get("eval_vy_mps", self._safe_longitudinal_speed_mps()))
        min_ttc = max(0.0, float(info.get("eval_min_ttc", 10.0)))
        min_thw = max(0.0, float(info.get("eval_min_thw", 5.0)))
        jerk_x_mps3 = abs(float(info.get("eval_abs_jerk_x_mps3", 0.0)))
        jerk_y_mps3 = abs(float(info.get("eval_abs_jerk_y_mps3", info.get("eval_abs_jerk_mps3", 0.0))))
        jerk_mps3 = jerk_y_mps3
        goal_progress = float(np.clip(info.get("eval_goal_progress", 0.0), -1.0, 1.0))

        ttc_score = float(np.clip(min_ttc / 4.0, 0.0, 1.0))
        thw_score = float(np.clip(min_thw / max(self.thw_safe_seconds, 1e-6), 0.0, 1.0))
        eff_score = self._compute_eff_score(vy_mps)
        speed_over_norm = max(0.0, vy_mps - self.speed_over_mps) / max(self.speed_over_band_mps, 1e-6)
        speed_over_penalty = -float(np.clip(speed_over_norm ** 2, 0.0, 1.0))
        norm_x = jerk_x_mps3 / max(self.jerk_x_norm, 1e-6)
        norm_y = jerk_y_mps3 / max(self.jerk_y_norm, 1e-6)
        comfort_jerk2d = float(norm_x ** 2 + norm_y ** 2)
        comfort_score = -comfort_jerk2d

        safety_term = -1.0 if is_collided else (ttc_score - 1.0)
        # THW is still logged for diagnosis, but it no longer contributes reward shaping.
        thw_term = 0.0

        merge_bonus = self.merge_bonus if just_merged else 0.0
        success_bonus = self.success_bonus if is_endpoint_success and (terminated or truncated) else 0.0
        timeout_penalty = (
            self.timeout_penalty
            if (terminated or truncated) and (not is_endpoint_success) and (not self._has_collided)
            else 0.0
        )
        collision_penalty = self.collision_penalty if just_collided else 0.0

        return {
            "eff": eff_score,
            "safety": safety_term,
            "thw": thw_term,
            "comfort": comfort_score,
            "goal": goal_progress,
            "speed_over": speed_over_penalty,
            "merge_bonus": merge_bonus,
            "success_bonus": success_bonus,
            "timeout_penalty": timeout_penalty,
            "collision_penalty": collision_penalty,
            "merge_event": 1.0 if just_merged else 0.0,
            "success_event": 1.0 if success_bonus != 0.0 else 0.0,
            "timeout_event": 1.0 if timeout_penalty != 0.0 else 0.0,
            "collision_event": 1.0 if collision_penalty != 0.0 else 0.0,
            "min_ttc": min_ttc,
            "min_thw": min_thw,
            "jerk_x_mps3": jerk_x_mps3,
            "jerk_mps3": jerk_mps3,
            "jerk_y_mps3": jerk_y_mps3,
            "comfort_jerk2d": comfort_jerk2d,
            "vy_mps": vy_mps,
            "speed_over_excess_mps": max(0.0, vy_mps - self.speed_over_mps),
        }

    def _combine_terms(self, terms):
        reward_raw = (
            self.weights["eff"] * terms["eff"]
            + self.weights["safety"] * terms["safety"]
            + self.weights["thw"] * terms["thw"]
            + self.weights["comfort"] * terms["comfort"]
            + self.weights["goal"] * terms["goal"]
            + self.weights["speed_over"] * terms["speed_over"]
            + terms["merge_bonus"]
            + terms["success_bonus"]
            + terms["timeout_penalty"]
            + terms["collision_penalty"]
        )
        reward = float(np.clip(reward_raw, self.reward_clip_min, self.reward_clip_max))
        terms["reward_raw"] = float(reward_raw)
        terms["reward"] = reward
        return reward

    def _finalize_episode(self):
        steps = max(self._episode_steps, 1)

        def mean_term(name):
            return float(self._episode_term_sums.get(name, 0.0) / steps)

        summary = {
            "episodes_completed": 1.0,
            "train_reward_mean": float(self._episode_return),
            "train_episode_length_mean": float(self._episode_steps),
            "term_eff": mean_term("eff"),
            "term_safety": mean_term("safety"),
            "term_thw": mean_term("thw"),
            "term_comfort": mean_term("comfort"),
            "term_goal": mean_term("goal"),
            "term_speed_over": mean_term("speed_over"),
            "term_merge_bonus": mean_term("merge_bonus"),
            "term_success_bonus": mean_term("success_bonus"),
            "term_timeout_penalty": mean_term("timeout_penalty"),
            "term_collision_penalty": mean_term("collision_penalty"),
            "term_reward_raw": mean_term("reward_raw"),
            "term_reward_clipped": mean_term("reward"),
            "mean_min_ttc_train": mean_term("min_ttc"),
            "mean_min_thw_train": mean_term("min_thw"),
            "mean_abs_jerk_x_train": mean_term("jerk_x_mps3"),
            "mean_abs_jerk_train": mean_term("jerk_mps3"),
            "mean_abs_jerk_y_train": mean_term("jerk_y_mps3"),
            "mean_comfort_jerk2d_train": mean_term("comfort_jerk2d"),
            "mean_speed_train_mps": mean_term("vy_mps"),
            "merge_bonus_rate": float(self._episode_event_counts.get("merge_event", 0.0) / steps),
            "success_bonus_rate": float(self._episode_event_counts.get("success_event", 0.0) / steps),
            "timeout_penalty_rate": float(self._episode_event_counts.get("timeout_event", 0.0) / steps),
            "collision_penalty_rate": float(self._episode_event_counts.get("collision_event", 0.0) / steps),
        }
        self.completed_episode_summaries.append(summary)
        self._reset_episode_trackers()

    def pop_completed_episode_summaries(self):
        summaries = list(self.completed_episode_summaries)
        self.completed_episode_summaries.clear()
        return summaries
