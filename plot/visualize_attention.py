import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.collections import LineCollection
import pandas as pd
import os
import sys
import glob
from stable_baselines3 import PPO

# ==========================================
# 1. 路径与配置
# ==========================================
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from configs.config import Config
from envs.merging_env import MergingEnv
from utils.data_loader import MergingDataset
# 【极其关键】：必须导入我们写的网络结构，否则 PPO 无法从 zip 反序列化自定义架构
from model.attention_net import AttentionFeaturesExtractor, AttentionRewardNet

# 设置中文字体
try:
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
except:
    print("警告: 未找到 SimHei 字体，将使用默认字体")

def run_inference_and_log(model, cfg, data_path, expert_mean, expert_std):
    """
    运行推理，记录轨迹和注意力详情 (彻底对齐官方环境逻辑)
    """
    print(f"... 正在使用底层解析器处理测试文件: {os.path.basename(data_path)}")
    
    # 1. 用官方 Dataset 完美解析测试轨迹，保证物理状态计算与训练时 100% 一致！
    test_dataset = MergingDataset([data_path], device='cpu')
    if len(test_dataset) == 0:
        raise ValueError("无法解析该轨迹文件")
    
    # 2. 【核心修复】用全局训练集的统计量覆盖单一文件的统计量，保证归一化基准一致
    test_dataset.expert_mean = expert_mean
    test_dataset.expert_std = expert_std
    
    # 3. 构建标准环境
    env = MergingEnv(test_dataset)
    obs, _ = env.reset(seed=42) # reset 会自动装载 test_dataset 里的轨迹

    # ========================================================
    # 🌟 植入 PyTorch Hook 拦截 Attention 权重输出 
    # ========================================================
    captured_attn_weights = []
    def attn_hook(module, inp, out):
        captured_attn_weights.append(out[1].detach().cpu().numpy())
    
    # 挂载窃听器
    hook_handle = model.policy.features_extractor.attention.attn.register_forward_hook(attn_hook)

    traj_points = [] 
    logs = []        
    
    # 记录初始点
    px, py, vx, vy = env.ego_state
    traj_points.append([px, py, np.sqrt(vx**2 + vy**2)])

    done = False
    step_count = 0
    max_steps = 500

    while not done and step_count < max_steps:
        captured_attn_weights.clear() 
        
        # SB3 标准推理模式
        action, _ = model.predict(obs, deterministic=True)
        
        if len(captured_attn_weights) > 0:
            w_np = captured_attn_weights[0]
            weights = w_np[0, 0] if w_np.ndim == 3 else w_np[0]
        else:
            weights = np.zeros(3)

        # 反归一化获取真实物理坐标
        raw_state = obs[:16] * expert_std + expert_mean

        log_entry = {
            'Step': step_count,
            'Ego_X': raw_state[0],
            'Ego_Y': raw_state[1],
            'L6_Lead_Attn': weights[0],
            'L5_Lead_Attn': weights[1],
            'L5_Foll_Attn': weights[2]
        }
        logs.append(log_entry)

        # 环境步进
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        px, py, vx, vy = env.ego_state
        traj_points.append([px, py, np.sqrt(vx**2 + vy**2)])
        
        step_count += 1

    hook_handle.remove() 
    return np.array(traj_points), pd.DataFrame(logs)

def visualize_attention_analysis():
    print("=" * 80)
    print("开始注意力权重可视化分析 (对齐 Batch Env 逻辑)")
    print("=" * 80)

    cfg = Config()
    
    # ---------------------------------------------------------
    # 1. 获取全局归一化统计量 (与训练集严格保持一致)
    # ---------------------------------------------------------
    print("... 正在加载训练数据集以获取全局归一化基准 ...")
    stats_data_paths = [
        os.path.join(root_dir, 'data', 'lane_change_trajectories-0750am-0805am'),
        os.path.join(root_dir, 'data', 'lane_change_trajectories-0805am-0820am'),
        os.path.join(root_dir, 'data', 'lane_change_trajectories-0820am-0835am')
    ]
    train_dataset = MergingDataset(stats_data_paths, device='cpu')
    global_expert_mean = train_dataset.expert_mean
    global_expert_std = train_dataset.expert_std
    print("✅ 全局归一化器初始化完成")

    # ---------------------------------------------------------
    # 2. 加载模型
    # ---------------------------------------------------------
    try:
        # 修改为最新的 Attention 模型名称
        model_path = os.path.join(root_dir, "checkpoints", "baseline_policy_attn_epoch_200.zip")
        
        if not os.path.exists(model_path):
            zips = glob.glob(os.path.join(root_dir, "checkpoints", "*_attn_*.zip"))
            if zips:
                model_path = sorted(zips, key=os.path.getmtime)[-1]
        
        model = PPO.load(model_path, device="cpu")
        print(f"✅ 成功加载模型: {model_path}")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # ---------------------------------------------------------
    # 3. 选择测试文件
    # ---------------------------------------------------------
    target_filename = "vehicle_1280_trajectory.csv"
    data_root = os.path.join(root_dir, "data")
    files = glob.glob(os.path.join(data_root, "**", target_filename), recursive=True)
    
    if not files:
        files = glob.glob(os.path.join(data_root, "**", "*.csv"), recursive=True)
        if not files: return
    
    data_path = files[0]
    print(f"📂 读取测试文件: {data_path}")

    # 读取 GT 仅用于画图参考
    df_gt = pd.read_csv(data_path)
    gt_x, gt_y = df_gt['KF_Local_X'].values, df_gt['KF_Local_Y'].values
    
    # ---------------------------------------------------------
    # 4. 运行推理 (传入 data_path 和全局统计量)
    # ---------------------------------------------------------
    print("🧠 运行严格对齐的推理环境...")
    traj_points, log_df = run_inference_and_log(model, cfg, data_path, global_expert_mean, global_expert_std)

    # ---------------------------------------------------------
    # 5. 绘图部分
    # ---------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # --- 左图: 轨迹 ---
    model_x = traj_points[:, 0]
    model_y = traj_points[:, 1]
    model_v = traj_points[:, 2]

    vmin_val, vmax_val = 0.0, 80.0
    y_min, y_max = min(np.min(gt_y), np.min(model_y)) - 20, max(np.max(gt_y), np.max(model_y)) + 20

    boundary_style = {'color': 'gray', 'linestyle': '--', 'linewidth': 1.0, 'alpha': 0.7}
    ax1.vlines([cfg.X_MIN, cfg.X_MAX], y_min, y_max, **boundary_style)
    ax1.vlines([cfg.X_MIN + cfg.LANE_WIDTH], y_min, y_max, color='black', linestyle='--', linewidth=1.5, alpha=0.8)

    # 绘制 GT 和 Model 轨迹
    ax1.plot(gt_x, gt_y, color='black', linewidth=1.5, alpha=0.5, label='Human Expert (GT)', zorder=2)
    sc = ax1.scatter(model_x, model_y, c=model_v, cmap='RdYlGn_r', s=25, edgecolors='black', linewidths=0.2, label='AI Model', zorder=3, vmin=vmin_val, vmax=vmax_val)
    
    ax1.set_title('Trajectory Comparison', fontsize=14, fontweight='bold')
    ax1.set_aspect(0.5)
    ax1.set_xlim(cfg.X_MIN - 5, cfg.X_MAX + 5)
    ax1.set_ylim(y_min, y_max)
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle=':', alpha=0.3)

    # --- 右图: 注意力柱状图与变道辅助线 ---
    attn_data = log_df[['L6_Lead_Attn', 'L5_Lead_Attn', 'L5_Foll_Attn']].values
    ego_y_data = log_df['Ego_Y'].values
    
    if len(ego_y_data) > 0:
        total_steps = len(ego_y_data)
        step_size = max(5, total_steps // 40)
        
        indices = np.arange(0, total_steps, step_size)
        sampled_y = ego_y_data[indices]
        sampled_w = attn_data[indices]
        
        bar_width = np.mean(np.diff(sampled_y)) * 0.8 if len(sampled_y) > 1 else 5.0

        ax2.bar(sampled_y, sampled_w[:, 0], width=bar_width, label='L6 Lead (Origin Front)', color='#FF9999', edgecolor='white')
        ax2.bar(sampled_y, sampled_w[:, 1], width=bar_width, bottom=sampled_w[:, 0], label='L5 Lead (Target Front)', color='#66B2FF', edgecolor='white')
        ax2.bar(sampled_y, sampled_w[:, 2], width=bar_width, bottom=sampled_w[:, 0]+sampled_w[:, 1], label='L5 Follow (Target Rear)', color='#99FF99', edgecolor='white')
        
        # 【新增：精确变道虚线】
        # 计算变道边界，自车从 L6 (~66) 变到 L5 (~54)，当横向坐标 X 跨越边界时即为变道发生
        divider_x = cfg.X_MIN + cfg.LANE_WIDTH
        crossed_boundary = log_df['Ego_X'] < divider_x 
        
        if crossed_boundary.any():
            merge_idx = crossed_boundary.idxmax() # 找到第一次跨过边界的索引
            merge_y = log_df.loc[merge_idx, 'Ego_Y']
            
            # 画红色虚线
            ax2.axvline(x=merge_y, color='red', linestyle='--', linewidth=2.5, zorder=5)
            # 添加文本标注
            ax2.text(merge_y, 1.05, "Lane Change Moment", ha='center', va='bottom', color='red', fontweight='bold', fontsize=12, zorder=5)

        ax2.set_title('Dynamic Social Attention Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Longitudinal Position (Y) [ft]')
        ax2.set_ylabel('Attention Weight Portion')
        ax2.set_ylim(0, 1.15) # 稍微调高一点给文字留空间
        ax2.set_xlim(min(ego_y_data)-10, max(ego_y_data)+10)
        ax2.legend(loc='upper right', title="Neighbor Vehicles")
        ax2.grid(True, axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    
    output_path = os.path.join(root_dir, "plot", "attention_analysis_aligned.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"✅ 绘图完成！精确对齐且含变道虚线的图片已保存为: {output_path}")
    plt.show()

if __name__ == "__main__":
    visualize_attention_analysis()