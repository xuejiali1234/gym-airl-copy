import torch
import numpy as np
import matplotlib

# 设置非交互式后端，防止弹窗
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import os
import sys
import random
import traceback
from stable_baselines3 import PPO

# ==========================================
# 1. 路径与配置
# ==========================================
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from configs.config import Config
from utils.data_loader import MergingDataset
from envs.merging_env import MergingEnv

# 设置中文字体
try:
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
except:
    print("警告: 未找到 SimHei 字体，将使用默认字体")


class SingleTrajDataset:
    """
    [新增组件] 单轨迹虚拟数据集包装器
    极其重要：用它直接包装已经完美归一化的轨迹，彻底避免读取单个 CSV 造成的局部归一化污染。
    """
    def __init__(self, traj, expert_mean, expert_std):
        self.trajectories = [traj]
        self.expert_mean = expert_mean
        self.expert_std = expert_std

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.trajectories[idx]


def run_inference(model, env):
    """
    运行模型推理，直接利用 Gym 环境进行仿真
    """
    model_x, model_y, model_v = [], [], []
    
    # 重置环境 (会自动加载 SingleTrajDataset 中的唯一轨迹)
    obs, info = env.reset()
    
    # 记录初始点物理状态
    px, py, vx, vy = env.ego_state
    model_x.append(px)
    model_y.append(py)
    model_v.append(np.sqrt(vx**2 + vy**2))

    done = False
    truncated = False
    max_steps = len(env.current_traj['ego_pos']) + 50 

    step_count = 0
    while not (done or truncated) and step_count < max_steps:
        # 使用 SB3 进行动作预测
        action, _states = model.predict(obs, deterministic=True)

        # 环境步进
        obs, reward, done, truncated, info = env.step(action)
        
        # 记录新物理状态
        px, py, vx, vy = env.ego_state
        model_x.append(px)
        model_y.append(py)
        model_v.append(np.sqrt(vx**2 + vy**2))
        
        step_count += 1

    return model_x, model_y, model_v


def visualize_trajectory_batch():
    print("=" * 80)
    print("开始批量轨迹可视化 (Random 20 Samples)")
    print("=" * 80)

    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------------------------------------
    # 1. 加载全局数据集与全局统计量
    # ---------------------------------------------------------
    stats_data_paths = [
        os.path.join(root_dir, 'data', 'lane_change_trajectories-0750am-0805am'),
        os.path.join(root_dir, 'data', 'lane_change_trajectories-0805am-0820am'),
        os.path.join(root_dir, 'data', 'lane_change_trajectories-0820am-0835am')
    ]
    
    print("... 正在加载全局数据集以获取完美归一化的轨迹 ...")
    try:
        global_dataset = MergingDataset(stats_data_paths, device=device)
        if len(global_dataset) == 0:
            raise ValueError("数据集为空!")
        expert_mean, expert_std = global_dataset.get_stats()
        print(f"✅ 全局数据集加载完成，共解析 {len(global_dataset)} 条有效轨迹")
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        return

    # ---------------------------------------------------------
    # 2. 加载模型
    # ---------------------------------------------------------
    try:
        model_path = os.path.join(root_dir, "checkpoints", "baseline_policy_attn_goal_zero_branch_noaux_epoch_300.zip") 
        if not os.path.exists(model_path):
            print(f"❌ 找不到模型文件: {model_path}，请修改为正确的 zip 路径。")
            return
            
        print(f"\n--- 正在加载 SB3 策略模型 ---")
        model = PPO.load(model_path, device=device)
        print(f"✅ 模型加载成功\n")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        traceback.print_exc()
        return

    # ---------------------------------------------------------
    # 3. 随机抽取轨迹进行推演
    # ---------------------------------------------------------
    num_samples = min(20, len(global_dataset))
    # 随机抽取轨迹索引
    selected_indices = random.sample(range(len(global_dataset)), num_samples)
    
    output_dir = os.path.join(root_dir, "plot", "batch_results")
    os.makedirs(output_dir, exist_ok=True)
    print(f"📂 结果将保存至: {output_dir}")

    for i, idx in enumerate(selected_indices):
        # 取出已经处理好的单条轨迹字典
        traj = global_dataset[idx]
        filename = traj.get('filename', f'trajectory_{idx}.csv')
        print(f"\n[{i + 1}/{num_samples}] 处理轨迹: {filename}")

        try:
            # --- A. 包装为单轨迹数据集并传入环境 ---
            single_dataset = SingleTrajDataset(traj, expert_mean, expert_std)
            env = MergingEnv(single_dataset)

            # --- B. 提取绝对正确的 Ground Truth (直接从数据加载器提取) ---
            gt_x = traj['ego_pos'][:, 0]
            gt_y = traj['ego_pos'][:, 1]
            gt_vx = traj['ego_vel'][:, 0]
            gt_vy = traj['ego_vel'][:, 1]
            # 计算真实速度大小
            gt_v = np.sqrt(gt_vx**2 + gt_vy**2)

            # --- C. 运行模型推理 ---
            model_x, model_y, model_v = run_inference(model, env)

            # --- D. 绘图 (逻辑不变) ---
            fig, ax = plt.subplots(figsize=(7, 8.5)) 

            vmin_val, vmax_val = 0.0, 80.0
            y_min, y_max = min(np.min(gt_y), np.min(model_y)) - 20, max(np.max(gt_y), np.max(model_y)) + 20

            # 绘制道路边界
            boundary_style = {'color': 'gray', 'linestyle': '--', 'linewidth': 1.0, 'alpha': 0.7}
            ax.vlines([cfg.X_MIN, cfg.X_MAX], y_min, y_max, **boundary_style)
            ax.vlines([cfg.X_MIN + cfg.LANE_WIDTH], y_min, y_max, color='black', linestyle='--', linewidth=1.5, alpha=0.8)

            # 绘制 Ground Truth 颜色渐变线
            points = np.array([gt_x, gt_y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap='RdYlGn_r', linewidth=4.0, alpha=0.7, zorder=2)
            lc.set_array(gt_v)
            lc.set_clim(vmin=vmin_val, vmax=vmax_val)
            ax.add_collection(lc)
            ax.plot(gt_x, gt_y, color='black', linewidth=1.0, alpha=0.4, label='Ground Truth', zorder=2)
            
            # 绘制 Model 轨迹散点
            sc = ax.scatter(model_x, model_y, c=model_v, cmap='RdYlGn_r', s=25, edgecolors='black', linewidths=0.5, label='GC-AIRL (Gym)', zorder=3, vmin=vmin_val, vmax=vmax_val)
            cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Velocity (ft/s)', rotation=270, labelpad=15)

            # 绘制起点和终点标记
            ax.scatter(gt_x[0], gt_y[0], c='green', s=120, label='Start', zorder=5, edgecolors='white')
            ax.scatter(gt_x[-1], gt_y[-1], c='red', marker='X', s=120, label='End (GT)', zorder=5, edgecolors='white')
            ax.scatter(model_x[-1], model_y[-1], c='purple', marker='*', s=200, label='End (Model)', zorder=6, edgecolors='white')

            # 图片修饰
            ax.set_title(f'Trajectory Comparison\n{filename}', fontsize=12, fontweight='bold', pad=25)
            ax.set_aspect(0.5)
            ax.set_xlim(cfg.X_MIN - 5, cfg.X_MAX + 5)
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel('Lateral Position (ft)')
            ax.set_ylabel('Longitudinal Position (ft)')
            ax.legend(loc='upper right', bbox_to_anchor=(-0.1, 1.0), fontsize=8)
            ax.grid(True, linestyle=':', alpha=0.3)

            save_path = os.path.join(output_dir, filename.replace('.csv', '.png'))
            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            plt.close(fig)
            print(f"   -> 已保存: {save_path}")

        except Exception as e:
            print(f"❌ 处理文件 {filename} 时发生错误: {str(e)}")
            traceback.print_exc()

    print("\n✅ 所有批量绘图任务完成！")


if __name__ == "__main__":
    visualize_trajectory_batch()