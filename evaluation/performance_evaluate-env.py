import torch
import numpy as np
import pandas as pd
import os
import sys
import glob
import random
from tqdm import tqdm
from stable_baselines3 import PPO

# ==========================================
# 1. 路径与配置
# ==========================================
curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(curr_dir)
sys.path.append(root_dir)

from configs.config import Config
from envs.merging_env import MergingEnv
from utils.data_loader import MergingDataset

class SingleTrajDataset:
    """
    单轨迹虚拟数据集包装器
    确保每个评估回合都能使用全局统一的完美归一化标准，防止污染
    """
    def __init__(self, traj, expert_mean, expert_std):
        self.trajectories = [traj]
        self.expert_mean = expert_mean
        self.expert_std = expert_std

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.trajectories[idx]


class MetricsEvaluator:
    def __init__(self):
        print("=" * 60)
        print("启动最终版物理评估 (Physics-Based Evaluator - Gym Version)")
        print("=" * 60)

        self.cfg = Config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ft_to_m = 0.3048  # 英尺转米

        # 1. 优先加载 Dataset 获取全局归一化统计量
        try:
            stats_data_paths = [
                os.path.join(root_dir, 'data', 'lane_change_trajectories-0750am-0805am'),
                os.path.join(root_dir, 'data', 'lane_change_trajectories-0805am-0820am'),
                os.path.join(root_dir, 'data', 'lane_change_trajectories-0820am-0835am')
            ]
            self.global_dataset = MergingDataset(stats_data_paths, device=self.device)
            if len(self.global_dataset) == 0:
                raise ValueError("加载轨迹为0，请检查路径！")
            
            self.expert_mean, self.expert_std = self.global_dataset.get_stats()
            
            print("✅ 全局归一化统计量获取完毕")
            print(f"📊 数据集加载完毕: {len(self.global_dataset)} 条轨迹")
        except Exception as e:
            print(f"❌ 数据集加载失败: {e}")
            self.global_dataset = []
            sys.exit(1)

        # 2. 加载 SB3 PPO 模型
        # 请根据实际情况确认你的 .zip 模型名称
        model_path = os.path.join(root_dir, "checkpoints", "baseline_policy_attn_goal_safe_branch_aux_probe_P300_D5_S06_seed44_epoch_290.zip")
        if not os.path.exists(model_path):
            print(f"❌ 未找到模型: {model_path}")
            sys.exit(1)

        try:
            print(f"--- 正在加载 SB3 策略模型 ---")
            self.model = PPO.load(model_path, device=self.device)
            print(f"✅ 模型加载成功: {model_path}")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            sys.exit(1)

    def evaluate(self, num_episodes=None):
        if not self.global_dataset or len(self.global_dataset) == 0:
            print("无法评估：数据集为空")
            return

        # 如果未指定评估数量，则默认评估整个数据集
        if num_episodes is None:
            num_episodes = len(self.global_dataset)

        indices = list(range(len(self.global_dataset)))
        if num_episodes < len(self.global_dataset):
            indices = random.sample(indices, num_episodes)

        stats = {
            'success': 0,
            'collision': 0,
            'timeout': 0,
            'total': 0
        }

        metrics_log = {
            'speeds': [],
            'jerks': [],
            'ttcs': [],
            'thws': []
        }

        print(f"🚀 开始评估 {len(indices)} 条轨迹...")

        for idx in tqdm(indices):
            # 取出全局数据集中处理好的单条轨迹字典
            traj = self.global_dataset[idx]
            
            # 使用包装器包裹轨迹并实例化独立环境
            single_dataset = SingleTrajDataset(traj, self.expert_mean, self.expert_std)
            env = MergingEnv(single_dataset)
            
            # Use the same collision threshold as the training environment.
            env.collision_margin = 1.0

            obs, info = env.reset()
            done = False
            truncated = False
            step_count = 0
            max_steps = len(env.current_traj['ego_pos']) + 50

            traj_speeds = []
            traj_jerks = []
            
            # 初始化物理速度状态
            px, py, vx, vy = env.ego_state
            last_v = np.sqrt(vx ** 2 + vy ** 2)
            last_acc = 0.0

            min_ttc = float('inf')
            min_thw = float('inf')

            episode_collided = False
            is_endpoint_success = False

            # === 物理推理循环 ===
            while not (done or truncated) and step_count < max_steps:
                # 动作推理 (完全交由 SB3 管理张量转换)
                action, _states = self.model.predict(obs, deterministic=True)
                
                # 环境推演
                obs, reward, done, truncated, info = env.step(action)
                episode_collided = episode_collided or bool(info.get('is_collided', False))

                # --- 收集物理数据 ---
                px, py, vx, vy = env.ego_state
                v_curr = np.sqrt(vx ** 2 + vy ** 2)

                acc_curr = (v_curr - last_v) / self.cfg.DT
                jerk_curr = abs(acc_curr - last_acc) / self.cfg.DT

                traj_speeds.append(v_curr)
                if step_count > 0:
                    traj_jerks.append(jerk_curr)

                last_v = v_curr
                last_acc = acc_curr

                # --- 计算 TTC / THW ---
                surr = env._get_surround_at_t(env.t)
                ego_front_y = py + self.cfg.VEHICLE_LENGTH / 2.0

                def check_vehicle_ttc(veh_idx):
                    base_idx = veh_idx * 4
                    if surr[base_idx] == 0 and surr[base_idx+1] == 0:
                        return float('inf'), float('inf')

                    target_rear_y = surr[base_idx+1] - self.cfg.VEHICLE_LENGTH / 2.0
                    dist = target_rear_y - ego_front_y
                    
                    if dist > 0 and abs(surr[base_idx] - px) < self.cfg.LANE_WIDTH:
                        rel_v = v_curr - surr[base_idx+2] 
                        
                        thw = dist / v_curr if v_curr > 1.0 else float('inf')
                        ttc = dist / rel_v if rel_v > 0.1 else float('inf')
                        return ttc, thw
                    return float('inf'), float('inf')
                
                ttc1, thw1 = check_vehicle_ttc(0) # L6 Lead
                ttc2, thw2 = check_vehicle_ttc(1) # L5 Lead
                
                min_ttc = min(min_ttc, ttc1, ttc2)
                min_thw = min(min_thw, thw1, thw2)

                step_count += 1

                # 记录回合结果
                if done or truncated:
                    is_endpoint_success = info.get('is_endpoint_success', info.get('is_success', False))
                    episode_collided = episode_collided or bool(getattr(env, 'has_collided_this_episode', False))

            # === 单条轨迹统计结束 ===
            stats['total'] += 1
            if is_endpoint_success:
                stats['success'] += 1
            if episode_collided:
                stats['collision'] += 1
            if not is_endpoint_success:
                stats['timeout'] += 1

            if traj_speeds:
                metrics_log['speeds'].append(np.mean(traj_speeds))
            if traj_jerks:
                metrics_log['jerks'].append(np.mean(traj_jerks))
            if min_ttc != float('inf'): metrics_log['ttcs'].append(min_ttc)
            if min_thw != float('inf'): metrics_log['thws'].append(min_thw)

        # === 打印最终报表 ===
        self.print_table(stats, metrics_log)

    def print_table(self, stats, metrics):
        total = stats['total']
        if total == 0: return

        print("\n" + "=" * 65)
        print(f"EVALUATION REPORT (N={total})")
        print("=" * 65)
        print(f"{'Metric':<25} | {'Value':<15} | {'Unit':<10}")
        print("-" * 65)

        # 1. 核心指标
        sr = stats['success'] / total
        cr = stats['collision'] / total
        tr = stats['timeout'] / total

        print(f"{'Success Rate':<25} | {sr:.3f}           | -")
        print(f"{'Collision Rate':<25} | {cr:.3f}           | -")
        print(f"{'Timeout/Stuck Rate':<25} | {tr:.3f}           | -")
        print("-" * 65)

        # 2. 物理指标 (转换为公制)
        avg_speed = np.mean(metrics['speeds']) * self.ft_to_m if metrics['speeds'] else 0
        avg_jerk = np.mean(metrics['jerks']) * self.ft_to_m if metrics['jerks'] else 0
        avg_ttc = np.mean(metrics['ttcs']) if metrics['ttcs'] else 0
        avg_thw = np.mean(metrics['thws']) if metrics['thws'] else 0

        print(f"{'Avg Speed':<25} | {avg_speed:.3f}           | m/s")
        print(f"{'Avg Jerk':<25} | {avg_jerk:.3f}           | m/s^3")
        print("-" * 65)
        print(f"{'Avg TTC (Min)':<25} | {avg_ttc:.3f}           | s")
        print(f"{'Avg THW (Min)':<25} | {avg_thw:.3f}           | s")
        print("=" * 65)


if __name__ == "__main__":
    evaluator = MetricsEvaluator()
    # 可以通过传入 num_episodes 来进行快速小批量测试，如: evaluator.evaluate(num_episodes=50)
    evaluator.evaluate()
