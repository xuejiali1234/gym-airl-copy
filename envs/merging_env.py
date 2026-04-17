import gymnasium as gym
import numpy as np
from gymnasium import spaces
from configs.config import Config

class MergingEnv(gym.Env):
    """
    基于 Gymnasium 重构的匝道合流环境
    """
    def __init__(self, dataset):
        super(MergingEnv, self).__init__()
        self.dataset = dataset
        self.cfg = Config()

        # [新增] 从数据加载器中提取专家统计量，用于实时归一化
        self.expert_mean = self.dataset.expert_mean
        self.expert_std = self.dataset.expert_std

        # 定义动作空间: [-1.0, 1.0]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # [修改点 1]: 动态决定状态维度 (18维 or 16维)
        self.enable_goal = getattr(self.cfg, 'ENABLE_GOAL_CONDITION', False)
        obs_dim = 18 if self.enable_goal else 16
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.current_traj = None
        self.t = 0
        self.ego_state = np.zeros(4, dtype=np.float32)
        self.start_y = 0.0

        self.has_merged = False
        self.collision_margin = 1.0
        self.has_collided_this_episode = False
        self.start_dist_to_goal = 1.0
        self.prev_ay_phys = 0.0
        self.eval_dense_return = 0.0

    def reset(self, seed=None, options=None):
        # 必须调用父类的 reset 来处理随机种子
        super().reset(seed=seed)
        
        # 使用提供的 seed 或 numpy 随机选择轨迹
        idx = int(self.np_random.integers(0, len(self.dataset)))
        
        self.current_traj = self.dataset[idx]
        self.t = 0

        init_pos = self.current_traj['ego_pos'][0].copy()
        init_vel = self.current_traj['ego_vel'][0].copy()

        self.ego_state = np.concatenate([init_pos, init_vel]).astype(np.float32)
        self.start_y = init_pos[1]
        goal_xy = self.current_traj['ego_pos'][-1]
        self.start_dist_to_goal = float(np.linalg.norm(self.ego_state[:2] - goal_xy))
        self.prev_ay_phys = 0.0
        self.eval_dense_return = 0.0

        self.has_merged = False
        self.has_collided_this_episode = False

        obs = self._get_obs()
        info = {}
        return obs, info

    def _get_surround_at_t(self, t):
        data = self.current_traj['surround_data']
        max_idx = len(data) - 1
        if t <= max_idx: return data[t]

        last_frame = data[max_idx]
        prev_frame = data[max(max_idx - 1, 0)]
        delta_time = (t - max_idx) * self.cfg.DT
        extrapolated = last_frame.copy()
        for i in range(0, 16, 4):
            if last_frame[i] == 0 and last_frame[i + 1] == 0: continue
            # Keep the tail-step neighbor vx consistent with data_loader.calc_vx(),
            # which repeats the last finite-difference value at the final frame.
            if max_idx > 0:
                last_vx = (last_frame[i] - prev_frame[i]) / self.cfg.DT
            else:
                last_vx = 0.0
            extrapolated[i] = last_frame[i] + last_vx * delta_time
            extrapolated[i + 1] = last_frame[i + 1] + last_frame[i + 2] * delta_time
            extrapolated[i + 3] = 0.0
        return extrapolated

    def _check_collision(self, px, py, surr_data):
        VEHICLE_LENGTH = 15.0
        VEHICLE_WIDTH = 6.0
        margin = self.collision_margin
        my_box = [px, py, VEHICLE_LENGTH * margin, VEHICLE_WIDTH * margin]

        for i in range(0, 12, 4):
            ox = surr_data[i]
            oy = surr_data[i + 1]
            if ox == 0 and oy == 0: continue

            dx = abs(px - ox)
            dy = abs(py - oy)

            if dx < (my_box[3] + VEHICLE_WIDTH * margin) / 2 and \
                    dy < (my_box[2] + VEHICLE_LENGTH * margin) / 2:
                return True
        return False

    def _compute_min_ttc_thw(self, px, py, vy, surr_data):
        ego_front_y = py + self.cfg.VEHICLE_LENGTH / 2.0
        ego_rear_y = py - self.cfg.VEHICLE_LENGTH / 2.0
        ttcs = []
        thws = []

        for base in (0, 4, 8):
            ox = surr_data[base]
            oy = surr_data[base + 1]
            ovy = surr_data[base + 2]
            if ox == 0 and oy == 0:
                continue
            if abs(ox - px) > self.cfg.LANE_WIDTH:
                continue

            if oy >= py:
                gap = (oy - self.cfg.VEHICLE_LENGTH / 2.0) - ego_front_y
                if gap > 0:
                    thws.append(gap / max(vy, 1e-3))
                    rel_v = vy - ovy
                    if rel_v > 0.1:
                        ttcs.append(gap / rel_v)
            else:
                rear_gap = ego_rear_y - (oy + self.cfg.VEHICLE_LENGTH / 2.0)
                if rear_gap > 0:
                    thws.append(rear_gap / max(ovy, 1e-3))
                    rel_v = ovy - vy
                    if rel_v > 0.1:
                        ttcs.append(rear_gap / rel_v)

        min_ttc = min(ttcs) if ttcs else 20.0
        min_thw = min(thws) if thws else 10.0
        return float(min_ttc), float(min_thw)

    def _compute_eval_dense_reward(
        self,
        prev_state,
        ay_phys,
        curr_state,
        surr_now,
        just_merged,
        is_endpoint_success,
        is_safety_success,
        is_collided_new,
    ):
        ft_to_m = 0.3048
        px_prev, _, _, _ = prev_state
        px, py, _, vy = curr_state
        goal_xy = self.current_traj['ego_pos'][-1]
        lane_target_x = self.cfg.X_MIN + self.cfg.LANE_WIDTH - 3.28

        d_prev = np.linalg.norm(prev_state[:2] - goal_xy)
        d_curr = np.linalg.norm(curr_state[:2] - goal_xy)
        p_goal = max(0.0, d_prev - d_curr) / max(self.start_dist_to_goal, 1.0)

        lane_gap_prev = max(0.0, px_prev - lane_target_x)
        lane_gap_curr = max(0.0, px - lane_target_x)
        p_lane = max(0.0, lane_gap_prev - lane_gap_curr) / max(self.cfg.LANE_WIDTH, 1e-6)

        vy_mps = vy * ft_to_m
        ay_mps2 = abs(ay_phys) * ft_to_m
        jerk_mps3 = abs(ay_phys - self.prev_ay_phys) / self.cfg.DT * ft_to_m
        min_ttc, min_thw = self._compute_min_ttc_thw(px, py, vy, surr_now)

        e_t = np.clip(vy_mps / 15.0, 0.0, 1.0)
        s_ttc = np.clip(min_ttc / 4.0, 0.0, 1.0)
        s_thw = np.clip(min_thw / 2.0, 0.0, 1.0)
        c_jerk = np.clip(jerk_mps3 / 3.0, 0.0, 1.0)
        c_acc = np.clip(ay_mps2 / 3.0, 0.0, 1.0)

        r_eval = 0.0
        r_eval += 0.5 if just_merged else 0.0
        r_eval += 2.0 * p_goal
        r_eval += 1.2 * p_lane
        r_eval += 0.020 * e_t
        r_eval += 0.015 * s_ttc
        r_eval += 0.010 * s_thw
        r_eval -= 0.015 * c_jerk
        r_eval -= 0.005 * c_acc
        r_eval -= 2.0 if is_collided_new else 0.0
        r_eval += 0.3 if is_endpoint_success else 0.0
        r_eval += 0.7 if is_safety_success else 0.0

        return float(r_eval), {
            'eval_min_ttc': float(min_ttc),
            'eval_min_thw': float(min_thw),
            'eval_goal_progress': float(p_goal),
            'eval_lane_progress': float(p_lane),
            'eval_vy_mps': float(vy_mps),
            'eval_abs_jerk_mps3': float(jerk_mps3),
        }

    def step(self, action):
        steer_max = getattr(self.cfg, 'PHYS_STEER_MAX', 8.0)
        acc_max = getattr(self.cfg, 'PHYS_ACC_MAX', 15.0)

        ax = action[0] * steer_max
        ay = action[1] * acc_max

        px, py, vx, vy = self.ego_state
        prev_state = np.array([px, py, vx, vy], dtype=np.float32)
        dt = self.cfg.DT

        vx_new = vx + ax * dt
        vy_new = vy + ay * dt

        speed_limit = getattr(self.cfg, 'SPEED_LIMIT', 80.0)
        current_speed = np.sqrt(vx_new ** 2 + vy_new ** 2)
        if current_speed > speed_limit:
            ratio = speed_limit / current_speed
            vx_new *= ratio
            vy_new *= ratio

        if vy_new < 0: vy_new = 0.0

        px_new = px + vx_new * dt
        py_new = py + vy_new * dt

        # 两车车道中心线
        lane_divider_x = self.cfg.X_MIN + self.cfg.LANE_WIDTH
        # 在辅助车道
        in_aux_lane = px_new > lane_divider_x
        MERGE_DEADLINE = 1350.0

        half_width = self.cfg.VEHICLE_WIDTH / 2.0
        wall_min = self.cfg.X_MIN + half_width
        wall_max = self.cfg.X_MAX - half_width

        if px_new < wall_min:
            px_new = wall_min
            if vx_new < 0: vx_new *= 0.1
        elif px_new > wall_max:
            px_new = wall_max
            if vx_new > 0: vx_new *= 0.1

        # 在正常规定范围内
        is_in_bounds = (px_new >= self.cfg.X_MIN) and (px_new <= self.cfg.X_MAX)

        self.ego_state = np.array([px_new, py_new, vx_new, vy_new], dtype=np.float32)
        self.t += 1

        surr_now = self._get_surround_at_t(self.t)
        is_collided = self._check_collision(px_new, py_new, surr_now)
        is_collided_new = is_collided and (not self.has_collided_this_episode)

        if is_collided:
            self.has_collided_this_episode = True

        terminated = False
        truncated = False
        expert_end_pos = self.current_traj['ego_pos'][-1]
        dist_to_end = np.sqrt((px_new - expert_end_pos[0]) ** 2 + (py_new - expert_end_pos[1]) ** 2)
        goal_tolerance = getattr(self.cfg, 'GOAL_TOLERANCE', 15.0)

        # 逻辑区分: Terminated (任务结束) vs Truncated (超时)
        if py_new > MERGE_DEADLINE and in_aux_lane:
            terminated = True
        if dist_to_end < goal_tolerance:
            terminated = True
            
        if self.t >= len(self.current_traj['ego_pos']) + 50:
            truncated = True

        lane_width = getattr(self.cfg, 'LANE_WIDTH', 12.0)
        divider_x = self.cfg.X_MIN + lane_width
        # 在目标车道
        in_target_lane = px_new < (divider_x - 3.28)

        r_goal = 0.0
        just_merged = False

        if is_in_bounds:
            if in_target_lane and not self.has_merged:
                if not self.has_collided_this_episode:
                    r_goal += 0.5
                    just_merged = True
                    self.has_merged = True

        is_merge_success = self.has_merged
        is_endpoint_success = is_in_bounds and (terminated or truncated) and \
                     (dist_to_end < goal_tolerance) and \
                     (not in_aux_lane)
        
        is_safety_success = is_in_bounds and (terminated or truncated) and \
                     (dist_to_end < goal_tolerance) and \
                     (not self.has_collided_this_episode) and \
                     (not in_aux_lane)

        if is_endpoint_success:
            r_goal += 0.5
        if is_safety_success:
            r_goal += 1.0

        eval_dense_reward, dense_info = self._compute_eval_dense_reward(
            prev_state=prev_state,
            ay_phys=ay,
            curr_state=self.ego_state,
            surr_now=surr_now,
            just_merged=just_merged,
            is_endpoint_success=is_endpoint_success,
            is_safety_success=is_safety_success,
            is_collided_new=is_collided_new,
        )
        self.eval_dense_return += eval_dense_reward
        self.prev_ay_phys = ay

        info = {
            'is_success': is_endpoint_success,
            'is_merge_success': is_merge_success,
            'is_endpoint_success': is_endpoint_success,
            'is_safety_success': is_safety_success,
            'is_collided': is_collided,
            'eval_dense_reward': float(eval_dense_reward),
            'eval_dense_return': float(self.eval_dense_return),
        }
        info.update(dense_info)
        return self._get_obs(), r_goal, terminated, truncated, info

    def _get_obs(self):
        px, py, vx, vy = self.ego_state
        surr_now = self._get_surround_at_t(self.t)
        surr_next = self._get_surround_at_t(self.t + 1)
        dt = self.cfg.DT

        def calculate_target_vx(idx_x):
            if surr_now[idx_x] == 0 or surr_next[idx_x] == 0: return 0.0
            return (surr_next[idx_x] - surr_now[idx_x]) / dt

        l6_lead_vx = calculate_target_vx(0)
        l5_lead_vx = calculate_target_vx(4)
        l5_foll_vx = calculate_target_vx(8)

        l6_lead = surr_now[0:4]
        l5_lead = surr_now[4:8]
        l5_foll = surr_now[8:12]

        def is_valid(veh):
            return not (veh[0] == 0 and veh[1] == 0)

        def get_rel(target, target_vx, ego_x, ego_y, ego_vx, ego_vy):
            return np.array([target[0] - ego_x, target[1] - ego_y, target_vx - ego_vx, target[2] - ego_vy], dtype=np.float32)

        r_l6 = get_rel(l6_lead, l6_lead_vx, px, py, vx, vy)
        r_l5_lead = get_rel(l5_lead, l5_lead_vx, px, py, vx, vy)
        r_l5_foll = get_rel(l5_foll, l5_foll_vx, px, py, vx, vy)

        raw_state = np.concatenate([np.array([px, py, vx, vy], dtype=np.float32), r_l6, r_l5_lead, r_l5_foll])
        
        if not is_valid(l6_lead): raw_state[4:8] = 0.0
        if not is_valid(l5_lead): raw_state[8:12] = 0.0
        if not is_valid(l5_foll): raw_state[12:16] = 0.0

        # 1. 对基础状态执行归一化
        normalized_state = (raw_state - self.expert_mean) / self.expert_std

        # ==========================================
        # 【核心修复：拯救 PyTorch Mask 失效 BUG】
        # 必须在归一化之后，把空车位的特征“强制”重新刷成 0.0！
        # 这样 attention_net.py 里的 1e-5 阈值检测才能顺利拦截它。
        # ==========================================
        if not is_valid(l6_lead): 
            normalized_state[4:8] = 0.0
        if not is_valid(l5_lead): 
            normalized_state[8:12] = 0.0
        if not is_valid(l5_foll): 
            normalized_state[12:16] = 0.0
        # ==========================================
        
        enable_goal = getattr(self.cfg, 'ENABLE_GOAL_CONDITION', False)

        if enable_goal:
            # 2. 提取目标信息
            t_idx = min(self.t, len(self.current_traj['goal']) - 1)
            # 注意：current_traj['goal'] 在 data_loader 中已经被归一化过了！
            current_goal = self.current_traj['goal'][t_idx] 
            
            # 3. 拼接已经归一化的 state 和已经归一化的 goal
            obs_18 = np.concatenate([normalized_state, current_goal], axis=-1)
            return obs_18.astype(np.float32)
        else:
            return normalized_state.astype(np.float32)
