import torch
import numpy as np
import pandas as pd
import os
import sys
import glob
import random
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from stable_baselines3 import PPO

# ==========================================
# 1. 路径与配置
# ==========================================
# 获取当前脚本所在目录 (evaluation)
curr_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (GC-AIRL-Merging_副本)
root_dir = os.path.dirname(curr_dir)
sys.path.append(root_dir)

from configs.config import Config
from utils.data_loader import MergingDataset
from envs.merging_env import MergingEnv

# 设置绘图风格
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    try:
        plt.style.use('seaborn-whitegrid')
    except OSError:
        plt.style.use('ggplot') # Fallback

try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

class SingleTrajDataset:
    """
    单轨迹虚拟数据集包装器，用于确保环境正确加载单条轨迹并使用全局归一化统计量
    """
    def __init__(self, traj, expert_mean, expert_std):
        self.trajectories = [traj]
        self.expert_mean = expert_mean
        self.expert_std = expert_std

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.trajectories[idx]


class Figure5Generator:
    def __init__(self):
        self.cfg = Config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ft_to_m = 0.3048  # 英尺转米

        # 1. 优先加载 Dataset 获取全局归一化统计量
        try:
            print("... 正在加载数据集以获取全局完美归一化轨迹 ...")
            stats_data_paths = [
                os.path.join(root_dir, 'data', 'lane_change_trajectories-0750am-0805am'),
                os.path.join(root_dir, 'data', 'lane_change_trajectories-0805am-0820am'),
                os.path.join(root_dir, 'data', 'lane_change_trajectories-0820am-0835am')
            ]
            self.global_dataset = MergingDataset(stats_data_paths, device=self.device)
            if len(self.global_dataset) == 0:
                raise ValueError("用于计算统计量的数据集为空!")
            
            self.expert_mean, self.expert_std = self.global_dataset.get_stats()
            print("✅ 全局归一化统计量加载完毕")
        except Exception as e:
            print(f"❌ 初始化归一化器失败: {e}")
            sys.exit(1)

        # 2. 加载 SB3 PPO 模型
        model_path = os.path.join(root_dir, "checkpoints", "baseline_policy_attn_goal_safe_branch_aux_probe_P300_D5_S06_seed44_epoch_290.zip")
        if not os.path.exists(model_path):
            print(f"⚠️ 未找到模型: {model_path}，请检查路径。")
            sys.exit(1)
            
        print(f"--- 正在加载 SB3 策略模型 ---")
        self.model = PPO.load(model_path, device=self.device)
        print(f"✅ 模型加载成功: {model_path}")
        
        # 参数
        self.target_lane_divider = self.cfg.X_MIN + getattr(self.cfg, 'LANE_WIDTH', 12.0)
        self.car_len = 15.0

    def _calculate_traj_score(self, df):
        """
        根据论文公式计算轨迹评分 (用于筛选 Top 20%)
        Score = (mean_vy - std_vy) + mean_TTC - 5 * mean_abs_jerk
        """
        vy = df['KF_Vel_Y']
        efficiency = vy.mean() - vy.std()

        if 'KF_Acc_Y' in df:
            acc_y = df['KF_Acc_Y'].values
        else:
            acc_y = np.diff(df['KF_Vel_Y'].values, prepend=df['KF_Vel_Y'].values[0]) / self.cfg.DT
            
        jerk = np.diff(acc_y) / self.cfg.DT
        comfort = np.abs(jerk).mean()

        if 'L6_Leading_Local_Y' in df:
            dist = df['L6_Leading_Local_Y'] - df['KF_Local_Y']
            rel_vel = df['KF_Vel_Y'] - df['L6_Leading_Vel']

            mask = (rel_vel > 0.1) & (dist > 0)

            if mask.sum() > 0:
                ttc = dist[mask] / rel_vel[mask]
                ttc = np.clip(ttc, 0, 20.0)
                safety = ttc.mean()
            else:
                safety = 20.0
        else:
             safety = 20.0

        raw_score = 1.0 * efficiency + 1.0 * safety - 5.0 * comfort
        return raw_score / 100.0

    def calculate_metrics_frame(self, df_or_dict, is_dict=False):
        """从一帧数据中提取指标 (用于构建分布)"""
        if not is_dict:
            py = df_or_dict['KF_Local_Y'].values
            vx = df_or_dict['KF_Vel_X'].values
            vy = df_or_dict['KF_Vel_Y'].values
            if 'KF_Acc_Y' in df_or_dict:
                ay = df_or_dict['KF_Acc_Y'].values
            else:
                # 缺失加速度列时，必须使用速度差分计算真实的纵向加速度 (a = dv/dt)
                ay = np.diff(vy, prepend=vy[0]) / self.cfg.DT

            px = df_or_dict['KF_Local_X'].values
            l6_lead_y = df_or_dict['L6_Leading_Local_Y'].values
            l6_lead_v = df_or_dict['L6_Leading_Vel'].values
            l5_lead_y = df_or_dict['L5_Leading_Local_Y'].values
            l5_lead_v = df_or_dict['L5_Leading_Vel'].values
        else:
            py = np.array(df_or_dict['py'])
            px = np.array(df_or_dict['px'])
            vx = np.array(df_or_dict['vx'])
            vy = np.array(df_or_dict['vy'])
            ay = np.array(df_or_dict['ay'])

            l6_lead_y = np.array(df_or_dict['l6_lead_y'])
            l6_lead_v = np.array(df_or_dict['l6_lead_v'])
            l5_lead_y = np.array(df_or_dict['l5_lead_y'])
            l5_lead_v = np.array(df_or_dict['l5_lead_v'])

        speed = np.sqrt(vx ** 2 + vy ** 2) * self.ft_to_m
        acc = ay * self.ft_to_m

        ttc_list = []
        thw_list = []

        for i in range(len(py)):
            lead_y, lead_v = 0, 0
            if px[i] > self.target_lane_divider:  # Lane 6
                lead_y, lead_v = l6_lead_y[i], l6_lead_v[i]
            else:  # Lane 5
                lead_y, lead_v = l5_lead_y[i], l5_lead_v[i]

            if lead_y != 0:
                dist = lead_y - py[i] - self.car_len
                rel_v = vy[i] - lead_v

                if rel_v > 0.1 and dist > 0:
                    val = dist / rel_v
                    if val < 20: ttc_list.append(val)

                if vy[i] > 1.0 and dist > 0:
                    val = dist / vy[i]
                    if val < 10: thw_list.append(val)

        return speed, acc, ttc_list, thw_list

    def collect_data(self, n_samples=100):
        print(f"🔄 正在收集数据 (样本数: {n_samples})...")
        all_traj_infos = [] 

        n_samples = min(n_samples, len(self.global_dataset))
        selected_indices = random.sample(range(len(self.global_dataset)), n_samples)

        print("1️⃣ 处理真实轨迹并计算评分...")
        for idx in tqdm(selected_indices):
            try:
                traj = self.global_dataset[idx]
                filename = traj.get('filename', f'trajectory_{idx}.csv')
                
                # 重新构建用于评分和指标计算的 DataFrame (使用原始物理值)
                ego_pos = traj['ego_pos']
                ego_vel = traj['ego_vel']
                surround_data = traj['surround_data']
                
                df_dict = {
                    'KF_Local_X': ego_pos[:, 0], 'KF_Local_Y': ego_pos[:, 1],
                    'KF_Vel_X': ego_vel[:, 0], 'KF_Vel_Y': ego_vel[:, 1],
                    'L6_Leading_Local_X': surround_data[:, 0], 'L6_Leading_Local_Y': surround_data[:, 1], 'L6_Leading_Vel': surround_data[:, 2],
                    'L5_Leading_Local_X': surround_data[:, 4], 'L5_Leading_Local_Y': surround_data[:, 5], 'L5_Leading_Vel': surround_data[:, 6]
                }
                df = pd.DataFrame(df_dict)
                
                score = self._calculate_traj_score(df)
                s, a, t_list, h_list = self.calculate_metrics_frame(df, is_dict=False)
                
                all_traj_infos.append({
                    'traj': traj,
                    'df': df,
                    'score': score,
                    'metrics': {'speed': s, 'acc': a, 'ttc': t_list, 'thw': h_list}
                })
            except Exception as e:
                print(f"Error processing index {idx}: {e}")
                continue

        all_traj_infos.sort(key=lambda x: x['score'], reverse=True)
        n_top20 = max(1, int(len(all_traj_infos) * 0.2))
        top20_infos = all_traj_infos[:n_top20]
        
        print(f"✅ 已筛选 Top 20% 轨迹: {len(top20_infos)} / {len(all_traj_infos)}")

        real_data_all = {'speed': [], 'acc': [], 'ttc': [], 'thw': []}
        real_data_top20 = {'speed': [], 'acc': [], 'ttc': [], 'thw': []}

        for info in all_traj_infos:
            m = info['metrics']
            real_data_all['speed'].extend(m['speed'])
            real_data_all['acc'].extend(m['acc'])
            real_data_all['ttc'].extend(m['ttc'])
            real_data_all['thw'].extend(m['thw'])
            
        for info in top20_infos:
            m = info['metrics']
            real_data_top20['speed'].extend(m['speed'])
            real_data_top20['acc'].extend(m['acc'])
            real_data_top20['ttc'].extend(m['ttc'])
            real_data_top20['thw'].extend(m['thw'])

        print("2️⃣ 运行模型仿真...")
        model_data = {'speed': [], 'acc': [], 'ttc': [], 'thw': []}
        
        for info in tqdm(all_traj_infos):
            traj = info['traj']
            
            # 使用包装器包裹轨迹并实例化独立环境
            single_dataset = SingleTrajDataset(traj, self.expert_mean, self.expert_std)
            env = MergingEnv(single_dataset)

            sim_res = {'px': [], 'py': [], 'vx': [], 'vy': [], 'ay': [],
                       'l6_lead_y': [], 'l6_lead_v': [], 'l5_lead_y': [], 'l5_lead_v': []}

            obs, _ = env.reset()
            done, truncated = False, False
            max_steps = len(traj['ego_pos']) + 50 
            step_count = 0

            # 开始仿真
            while not (done or truncated) and step_count < max_steps:
                # 记录前一帧的速度用于计算加速度
                _, _, _, prev_vy = env.ego_state

                # 动作推理
                action, _states = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, _ = env.step(action)
                
                # 获取环境信息
                px, py, vx, vy = env.ego_state
                ay = (vy - prev_vy) / self.cfg.DT

                surr_now = env._get_surround_at_t(env.t)
                
                sim_res['px'].append(px); sim_res['py'].append(py)
                sim_res['vx'].append(vx); sim_res['vy'].append(vy); sim_res['ay'].append(ay)
                sim_res['l6_lead_y'].append(surr_now[1]); sim_res['l6_lead_v'].append(surr_now[2])
                sim_res['l5_lead_y'].append(surr_now[5]); sim_res['l5_lead_v'].append(surr_now[6])
                
                step_count += 1

            # 计算生成轨迹的 Metrics
            ms, ma, mt, mh = self.calculate_metrics_frame(sim_res, is_dict=True)
            model_data['speed'].extend(ms)
            model_data['acc'].extend(ma)
            model_data['ttc'].extend(mt)
            model_data['thw'].extend(mh)

        return real_data_all, real_data_top20, model_data

    def plot(self, real_data_all, real_data_top20, model_data):
        print("📊 正在绘图...")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        color_real = 'gray'
        color_top20 = 'green'
        color_model = 'red'
        alpha = 0.5
        bins = 30

        # (a) Speed
        ax = axes[0, 0]
        sns.histplot(real_data_all['speed'], color=color_real, label='Real Data (All)', kde=True, ax=ax, stat="density", bins=bins, alpha=alpha)
        sns.histplot(real_data_top20['speed'], color=color_top20, label='Real Data (Top 20%)', kde=True, ax=ax, stat="density", bins=bins, alpha=0.3)
        sns.histplot(model_data['speed'], color=color_model, label='GC-AIRL', kde=True, ax=ax, stat="density", bins=bins, alpha=alpha)
        ax.set_title('(a) Speed Distribution (m/s)')
        ax.set_xlabel('Speed (m/s)')
        ax.legend()

        # (b) Acceleration
        ax = axes[0, 1]
        sns.histplot(real_data_all['acc'], color=color_real, label='Real Data (All)', kde=True, ax=ax, stat="density", bins=bins, alpha=alpha)
        sns.histplot(real_data_top20['acc'], color=color_top20, label='Real Data (Top 20%)', kde=True, ax=ax, stat="density", bins=bins, alpha=0.3)
        sns.histplot(model_data['acc'], color=color_model, label='GC-AIRL', kde=True, ax=ax, stat="density", bins=bins, alpha=alpha)
        ax.set_title('(b) Acceleration Distribution (m/s²)')
        ax.set_xlabel('Acceleration (m/s²)')
        ax.set_xlim(-4, 4)
        ax.legend()

        # (c) TTC
        ax = axes[1, 0]
        sns.kdeplot(real_data_all['ttc'], color=color_real, label='Real Data (All)', fill=True, ax=ax, alpha=0.2)
        sns.kdeplot(real_data_top20['ttc'], color=color_top20, label='Real Data (Top 20%)', fill=True, ax=ax, alpha=0.2)
        sns.kdeplot(model_data['ttc'], color=color_model, label='GC-AIRL', fill=True, ax=ax, alpha=0.2)
        ax.set_title('(c) TTC Distribution (s)')
        ax.set_xlabel('Time to Collision (s)')
        ax.set_xlim(0, 15)
        ax.legend()

        # (d) Time Headway
        ax = axes[1, 1]
        sns.kdeplot(real_data_all['thw'], color=color_real, label='Real Data (All)', fill=True, ax=ax, alpha=0.2)
        sns.kdeplot(real_data_top20['thw'], color=color_top20, label='Real Data (Top 20%)', fill=True, ax=ax, alpha=0.2)
        sns.kdeplot(model_data['thw'], color=color_model, label='GC-AIRL', fill=True, ax=ax, alpha=0.2)
        ax.set_title('(d) Time Headway Distribution (s)')
        ax.set_xlabel('Time Headway (s)')
        ax.set_xlim(0, 5)
        ax.legend()

        plt.tight_layout()
        save_path = os.path.join(curr_dir, 'Figure5_Distribution_Comparison.png')
        plt.savefig(save_path, dpi=300)
        print(f"✅ Figure 5 已保存至: {save_path}")
        plt.show()

    def save_data(self, real_data_all, real_data_top20, model_data):
        print("💾 正在保存分布数据...")
        rows = []
        sources = {'Real_All': real_data_all, 'Real_Top20': real_data_top20, 'GC_AIRL': model_data}
        
        for source_name, data_dict in sources.items():
            for metric_name, values in data_dict.items():
                for v in values:
                    rows.append({'source': source_name, 'metric': metric_name, 'value': v})
        
        df = pd.DataFrame(rows)
        save_path = os.path.join(curr_dir, 'distribution_comparison_data.csv')
        df.to_csv(save_path, index=False)
        print(f"✅ 分布对比数据已保存至: {save_path}")


if __name__ == "__main__":
    gen = Figure5Generator()
    real_all, real_top20, model = gen.collect_data(n_samples=217)  
    gen.save_data(real_all, real_top20, model)
    gen.plot(real_all, real_top20, model)