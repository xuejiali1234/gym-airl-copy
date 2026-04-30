import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import glob
import os
from configs.config import Config


class MergingDataset(Dataset):
    """
    数据加载器 (Physical Action Normalization Version)
    核心修改:
    - 动作 (Action) 不再使用 MinMaxScaler 的统计归一化结果。
    - 改为使用物理极限进行固定比例缩放 (Physical Normalization)。
    - 专家动作 3.0 ft/s^2 -> 3.0 / 15.0 = 0.2 (网络输入)，而非之前的 1.0 (统计最大值)。
    """

    def _calculate_score(self, df):
        """
        根据论文公式计算轨迹评分 (用于 Ranking Loss)
        Score = (mean_vy - std_vy) + mean_TTC - 5 * mean_abs_jerk
        Paper Ref: Section 5.1
        """
        # 1. Efficiency: 纵向平均速度 - 纵向速度标准差
        vy = df['KF_Vel_Y']
        efficiency = vy.mean() - vy.std()

        # 2. Comfort: Jerk (加加速度) 的绝对值平均
        # Jerk = d(Acc) / dt, DT = 0.1
        acc_y = df['KF_Acc_Y'].values
        jerk = np.diff(acc_y) / 0.1
        comfort = np.abs(jerk).mean()

        # 3. Safety: 简化的 TTC (Time to Collision)
        # 前车: L6_Leading (目标车道)
        dist = df['L6_Leading_Local_Y'] - df['KF_Local_Y']
        rel_vel = df['KF_Vel_Y'] - df['L6_Leading_Vel']

        mask = (rel_vel > 0.1) & (dist > 0)

        if mask.sum() > 0:
            ttc = dist[mask] / rel_vel[mask]
            ttc = np.clip(ttc, 0, 20.0)  # 截断，20秒以上视为同样安全
            safety = ttc.mean()
        else:
            safety = 20.0  # 无前车或不接近，给最大安全分

        # 总分公式 (Weights: 1.0, 1.0, -5.0)
        raw_score = 1.0 * efficiency + 1.0 * safety - 5.0 * comfort

        # 归一化缩放 (保持在合理量级)
        return raw_score / 100.0

    def _process_data(self, df, filename="unknown"):
        if len(df) < 10: return

        # ==========================================
        # 1. [Refactor] Read Raw State & Calculate Relative State
        # ==========================================
        # Ego State
        ego_x = df['KF_Local_X'].values.astype(np.float32)
        ego_y = df['KF_Local_Y'].values.astype(np.float32)
        ego_vx = df['KF_Vel_X'].values.astype(np.float32)
        ego_vy = df['KF_Vel_Y'].values.astype(np.float32)

        # Helper to get surround vehicle data (handle missing columns safely)
        def get_surround_raw(prefix):
            # X, Y, Vel(Vy), Acc
            if f'{prefix}_Local_X' not in df.columns:
                l = len(df)
                return np.zeros(l), np.zeros(l), np.zeros(l), np.zeros(l)
            
            x = df[f'{prefix}_Local_X'].values.astype(np.float32)
            y = df[f'{prefix}_Local_Y'].values.astype(np.float32)
            vy = df[f'{prefix}_Vel'].values.astype(np.float32) # CSV 'Vel' is mostly Longitudinal Speed
            acc = df[f'{prefix}_Acc'].values.astype(np.float32)
            return x, y, vy, acc

        # Load Surround Vehicles
        l6_lead_x, l6_lead_y, l6_lead_vy, _ = get_surround_raw('L6_Leading')
        l5_lead_x, l5_lead_y, l5_lead_vy, _ = get_surround_raw('L5_Leading')
        l5_foll_x, l5_foll_y, l5_foll_vy, _ = get_surround_raw('L5_Following')

        # Calculate Vx for Surround Vehicles (using position difference)
        # Vx = (X_t+1 - X_t) / dt
        # Fill last element with 0 or previous value
        dt = 0.1
        def calc_vx(x_arr):
            vx = np.diff(x_arr) / dt
            return np.append(vx, vx[-1]) # Repeat last value to keep shape

        l6_lead_vx = calc_vx(l6_lead_x)
        l5_lead_vx = calc_vx(l5_lead_x)
        l5_foll_vx = calc_vx(l5_foll_x)

        # Handle '0' values (missing vehicles) -> Relative State should be 0 or indicator?
        # Current logic in Env: if valid, Rel; if invalid, set specific mask.
        # But here we just compute Raw Relative first.
        # If target is 0,0, relative will be large negative/positive. 
        # But let's follow Env's post-processing logic later? 
        # No, we must compute raw relative state here.
        
        # Relative State Calculation: Target - Ego
        def get_rel(tx, ty, tvx, tvy, ex, ey, evx, evy):
            # Mask for invalid targets (where X=0, Y=0)
            mask = (tx != 0) | (ty != 0)
            
            rx = np.zeros_like(tx)
            ry = np.zeros_like(ty)
            rvx = np.zeros_like(tvx)
            rvy = np.zeros_like(tvy)
            
            rx[mask] = tx[mask] - ex[mask]
            ry[mask] = ty[mask] - ey[mask]
            rvx[mask] = tvx[mask] - evx[mask]
            rvy[mask] = tvy[mask] - evy[mask]
            
            return rx, ry, rvx, rvy

        r_l6_x, r_l6_y, r_l6_vx, r_l6_vy = get_rel(l6_lead_x, l6_lead_y, l6_lead_vx, l6_lead_vy, ego_x, ego_y, ego_vx, ego_vy)
        r_l5_lead_x, r_l5_lead_y, r_l5_lead_vx, r_l5_lead_vy = get_rel(l5_lead_x, l5_lead_y, l5_lead_vx, l5_lead_vy, ego_x, ego_y, ego_vx, ego_vy)
        r_l5_foll_x, r_l5_foll_y, r_l5_foll_vx, r_l5_foll_vy = get_rel(l5_foll_x, l5_foll_y, l5_foll_vx, l5_foll_vy, ego_x, ego_y, ego_vx, ego_vy)

        # Construct 16-dim Raw State
        # 0-3: Ego
        # 4-7: L6 Lead
        # 8-11: L5 Lead
        # 12-15: L5 Follow
        raw_states = np.column_stack([
            ego_x, ego_y, ego_vx, ego_vy,
            r_l6_x, r_l6_y, r_l6_vx, r_l6_vy,
            r_l5_lead_x, r_l5_lead_y, r_l5_lead_vx, r_l5_lead_vy,
            r_l5_foll_x, r_l5_foll_y, r_l5_foll_vx, r_l5_foll_vy
        ])

        # ==========================================
        # 2. [核心修改] 读取 Action 并进行"物理归一化"
        # ==========================================
        # 我们不再读取 'KF_Acc_X_norm'，而是读取原始物理值 'KF_Acc_X'
        # 然后手动除以物理极限

        # 获取物理加速度
        phys_acc_x = df['KF_Acc_X'].values.astype(np.float32)  # 横向
        phys_acc_y = df['KF_Acc_Y'].values.astype(np.float32)  # 纵向

        # 手动归一化: Action = Phys / Max_Limit
        # 这样专家的 3.0 只会变成 0.2，而不是 1.0
        # 确保使用 Config 中定义的物理极限
        steer_limit = getattr(self.cfg, 'PHYS_STEER_MAX', 5.0)
        acc_limit = getattr(self.cfg, 'PHYS_ACC_MAX', 15.0)

        act_x = phys_acc_x / steer_limit
        act_y = phys_acc_y / acc_limit

        # 截断到 [-1, 1] 之间，防止异常值溢出
        act_x = np.clip(act_x, -1.0, 1.0)
        act_y = np.clip(act_y, -1.0, 1.0)

        actions = np.column_stack([act_x, act_y])

        # ==========================================

        # 3. 计算 Goal (基于轨迹终点)
        final_state = raw_states[-1] # Raw Goal
        goal_vec = np.array([final_state[0], final_state[1]], dtype=np.float32)
        goals = np.tile(goal_vec, (len(raw_states), 1))

        # 4. 准备 Env 需要的物理数据
        def get_col(name, default=0.0):
            return df[name].values if name in df.columns else np.full(len(df), default)

        surround_data = np.column_stack([
            get_col('L6_Leading_Local_X'), get_col('L6_Leading_Local_Y'), get_col('L6_Leading_Vel'),
            get_col('L6_Leading_Acc'),
            get_col('L5_Leading_Local_X'), get_col('L5_Leading_Local_Y'), get_col('L5_Leading_Vel'),
            get_col('L5_Leading_Acc'),
            get_col('L5_Following_Local_X'), get_col('L5_Following_Local_Y'), get_col('L5_Following_Vel'),
            get_col('L5_Following_Acc'),
            get_col('L6_Following_Local_X'), get_col('L6_Following_Local_Y'), get_col('L6_Following_Vel'),
            get_col('L6_Following_Acc')
        ])

        # Next States
        next_states = np.roll(raw_states, -1, axis=0)
        next_states[-1] = raw_states[-1]

        # 计算真实评分
        traj_score = self._calculate_score(df)

        self.trajectories.append({
            'filename': filename,
            'state': raw_states, # Temporarily store Raw States
            'action': actions,  # 已经是物理归一化后的动作
            'goal': goals,      # Raw Goals (Matches State)
            'next_state': next_states, # Raw Next States
            'score': traj_score,
            'ego_pos': df[['KF_Local_X', 'KF_Local_Y']].values,
            'ego_vel': df[['KF_Vel_X', 'KF_Vel_Y']].values,
            'surround_data': surround_data
        })

    def _calculate_and_apply_stats(self):
        """
        [Refactored] Calculate Mean/Std from all loaded raw trajectories
        using a mask to ignore zero-padded values, then normalize the data.
        """
        if not self.trajectories:
            return

        all_states = np.vstack([t['state'] for t in self.trajectories])
        
        # --- Masked Statistics Calculation ---
        mask = np.abs(all_states) > 1e-6
        means = []
        stds = []
        for i in range(all_states.shape[1]):
            valid_data = all_states[:, i][mask[:, i]]
            if len(valid_data) > 1: # Need at least 2 points to have std
                means.append(valid_data.mean())
                stds.append(valid_data.std())
            else: # If column is all zeros or has only one value
                means.append(0.0)
                stds.append(1.0)
        
        self.expert_mean = np.array(means, dtype=np.float32)
        self.expert_std = np.array(stds, dtype=np.float32) + 1e-8 # Add epsilon for stability

        print(f"[DataLoader] Expert Data Stats (Masked):")
        # np.set_printoptions(precision=4, suppress=True)
        # print(f"  Mean: {self.expert_mean}")
        # print(f"  Std:  {self.expert_std}")

        # --- Normalize In-Place ---
        for t in self.trajectories:
            # 记录归一化前的原始状态，用于判断哪里是空车位 (全0)
            raw_states = t['state'].copy()
            raw_next_states = t['next_state'].copy()

            # 1. 执行全局归一化
            t['state'] = (t['state'] - self.expert_mean) / self.expert_std
            t['next_state'] = (t['next_state'] - self.expert_mean) / self.expert_std
            
            # 2. [核心修复]: 强制将原始为空的车位重新刷成 0.0
            for idx in range(len(t['state'])):
                # L6 Leading (Indices 4:8)
                if np.abs(raw_states[idx, 4:8]).sum() < 1e-5:
                    t['state'][idx, 4:8] = 0.0
                if np.abs(raw_next_states[idx, 4:8]).sum() < 1e-5:
                    t['next_state'][idx, 4:8] = 0.0
                    
                # L5 Leading (Indices 8:12)
                if np.abs(raw_states[idx, 8:12]).sum() < 1e-5:
                    t['state'][idx, 8:12] = 0.0
                if np.abs(raw_next_states[idx, 8:12]).sum() < 1e-5:
                    t['next_state'][idx, 8:12] = 0.0
                    
                # L5 Following (Indices 12:16)
                if np.abs(raw_states[idx, 12:16]).sum() < 1e-5:
                    t['state'][idx, 12:16] = 0.0
                if np.abs(raw_next_states[idx, 12:16]).sum() < 1e-5:
                    t['next_state'][idx, 12:16] = 0.0

            # Goals are Position X,Y (first 2 dims of state)
            g_mean = self.expert_mean[0:2]
            g_std = self.expert_std[0:2]
            t['goal'] = (t['goal'] - g_mean) / g_std

    def get_stats(self):
        return self.expert_mean, self.expert_std

    def __init__(self, data_path, device='cpu'):
        self.device = device
        self.trajectories = []
        self.cfg = Config()  # 加载配置以获取物理极限
        
        # Default stats (will be overwritten)
        self.expert_mean = np.zeros(16, dtype=np.float32)
        self.expert_std = np.ones(16, dtype=np.float32)

        # [修改] 支持单个路径或路径列表
        if isinstance(data_path, str):
            paths_to_search = [data_path]
        elif isinstance(data_path, list):
            paths_to_search = data_path
        else:
            raise TypeError(f"data_path 必须是字符串或列表, 但收到了 {type(data_path)}")

        all_files = []
        for path in paths_to_search:
            if os.path.isfile(path):
                all_files.append(path)
            elif os.path.isdir(path):
                # 递归搜索目录下的所有 .csv
                all_files.extend(glob.glob(os.path.join(path, "**", "*.csv"), recursive=True))
        
        # 去重并排序
        files = sorted(list(set(all_files)))
        
        if not files:
            print(f"⚠️ 警告: 在路径 {paths_to_search} 中未找到任何 .csv 文件。")

        print(f"[DataLoader] Found {len(files)} files in {len(paths_to_search)} specified path(s).")

        for csv_file in files:
            try:
                # 过滤掉 _normalized 文件夹中的文件
                if '_normalized' in csv_file:
                    continue
                filename = os.path.basename(csv_file)
                df = pd.read_csv(csv_file)
                self._process_data(df, filename)
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")

        print(f"[DataLoader] Loaded {len(self.trajectories)} trajectories.")
        
        # Perform Global Normalization
        if len(self.trajectories) > 0:
            self._calculate_and_apply_stats()
        
        self.confidence_weights = np.ones(len(self.trajectories), dtype=np.float32)

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        return self.trajectories[idx]

    def update_confidence_weights(self, new_weights):
        """
        更新置信度权重
        new_weights: 应该是 shape 为 (num_trajs,) 的数组
        """
        # 1. 确保它是 numpy 数组
        if isinstance(new_weights, list):
            new_weights = np.array(new_weights, dtype=np.float32)
        elif torch.is_tensor(new_weights):
            new_weights = new_weights.detach().cpu().numpy().astype(np.float32)
        else:
            new_weights = np.array(new_weights, dtype=np.float32)
        
        # 2. 检查维度 (防止变成标量)
        if new_weights.ndim == 0:
            # 如果是标量，扩展为与轨迹数量相同的数组
            new_weights = np.full(len(self.trajectories), new_weights, dtype=np.float32)
        
        # 3. 赋值
        self.confidence_weights = new_weights

    def get_all_start_states(self, indices=None):
        if indices is None:
            indices = range(len(self.trajectories))
            
        s = [self.trajectories[i]['state'][0] for i in indices]
        g = [self.trajectories[i]['goal'][0] for i in indices]
        sc = [self.trajectories[i]['score'] for i in indices]
        return {
            'state': torch.tensor(np.array(s), dtype=torch.float32).to(self.device),
            'goal': torch.tensor(np.array(g), dtype=torch.float32).to(self.device),
            'score': torch.tensor(np.array(sc), dtype=torch.float32).to(self.device)
        }

    def sample_expert_batch(self, batch_size, indices=None):
        if indices is None:
            pool = range(len(self.trajectories))
        else:
            pool = indices
            
        # 从指定的 pool 中随机选择索引
        selected_indices = np.random.choice(pool, batch_size)
        
        s, a, g, ns, b = [], [], [], [], []
        for i in selected_indices:
            traj = self.trajectories[i]
            t = np.random.randint(len(traj['state']))
            s.append(traj['state'][t])
            a.append(traj['action'][t])
            g.append(traj['goal'][t])
            ns.append(traj['next_state'][t])
            b.append(self.confidence_weights[i])

        return {
            'state': torch.tensor(np.array(s), dtype=torch.float32).to(self.device),
            'action': torch.tensor(np.array(a), dtype=torch.float32).to(self.device),
            'goal': torch.tensor(np.array(g), dtype=torch.float32).to(self.device),
            'next_state': torch.tensor(np.array(ns), dtype=torch.float32).to(self.device),
            'beta': torch.tensor(np.array(b), dtype=torch.float32).to(self.device)
        }
