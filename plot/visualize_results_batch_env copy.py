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

MODEL_TAG = "P30_CPairD250_NoLateLR_epoch298"
MODEL_PATH = os.path.join(
    root_dir,
    "train_log",
    "baseline_attn_goal_safe_branch_aux_probe_P30_CPairD250_NoLateLR_Save1_20260502_215110",
    "checkpoints",
    "baseline_policy_attn_goal_safe_branch_aux_probe_P30_CPairD250_NoLateLR_Save1_epoch_298.zip",
)

FT_TO_M = 0.3048

# Paper-style figure defaults.
matplotlib.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 10,
    "axes.unicode_minus": False,
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "legend.frameon": False,
})


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
    print("开始批量轨迹可视化 (All Trajectories)")
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
        print(f"[OK] 全局数据集加载完成，共解析 {len(global_dataset)} 条有效轨迹")
    except Exception as e:
        print(f"[ERROR] 数据集加载失败: {e}")
        return

    # ---------------------------------------------------------
    # 2. 加载模型
    # ---------------------------------------------------------
    try:
        model_path = MODEL_PATH
        if not os.path.exists(model_path):
            print(f"[ERROR] 找不到模型文件: {model_path}，请修改为正确的 zip 路径。")
            return
            
        print(f"\n--- 正在加载 SB3 策略模型 ---")
        print(f"模型标签: {MODEL_TAG}")
        print(f"模型路径: {model_path}")
        model = PPO.load(model_path, device=device)
        print(f"[OK] 模型加载成功\n")
    except Exception as e:
        print(f"[ERROR] 模型加载失败: {e}")
        traceback.print_exc()
        return

    # ---------------------------------------------------------
    # 3. 随机抽取轨迹进行推演
    # ---------------------------------------------------------
    target_filename = os.environ.get("VIS_TARGET_FILENAME", "").strip()
    if target_filename:
        target_basename = os.path.basename(target_filename)
        selected_indices = [
            idx for idx in range(len(global_dataset))
            if os.path.basename(str(global_dataset[idx].get("filename", ""))) == target_basename
        ]
        if not selected_indices:
            selected_indices = [
                idx for idx in range(len(global_dataset))
                if target_basename in os.path.basename(str(global_dataset[idx].get("filename", "")))
            ]
        if not selected_indices:
            print(f"[ERROR] 全局数据集中找不到指定轨迹: {target_filename}")
            return
        selected_indices = selected_indices[:1]
        num_samples = 1
    else:
        sample_spec = os.environ.get("VIS_NUM_SAMPLES", "all").strip().lower()
        if sample_spec in {"", "all", "*", "0"}:
            selected_indices = list(range(len(global_dataset)))
            num_samples = len(selected_indices)
        else:
            num_samples = min(int(sample_spec), len(global_dataset))
            rng = random.Random(int(os.environ.get("VIS_RANDOM_SEED", "42")))
            # 随机抽取轨迹索引
            selected_indices = rng.sample(range(len(global_dataset)), num_samples)
    
    print(f"[INFO] 本次绘制轨迹数: {num_samples}")
    
    output_tag = MODEL_TAG
    if not target_filename and len(selected_indices) == len(global_dataset):
        output_tag = f"{MODEL_TAG}_all"
    output_dir = os.path.join(root_dir, "plot", "batch_results", output_tag)
    os.makedirs(output_dir, exist_ok=True)
    print(f"[OUTPUT] 结果将保存至: {output_dir}")

    for i, idx in enumerate(selected_indices):
        # 取出已经处理好的单条轨迹字典
        traj = global_dataset[idx]
        filename = traj.get('filename', f'trajectory_{idx}.csv')
        print(f"\n[{i + 1}/{num_samples}] 处理轨迹: {filename}")

        try:
            # --- A. 包装为单轨迹数据集并传入环境 ---
            single_dataset = SingleTrajDataset(traj, expert_mean, expert_std)
            env = MergingEnv(single_dataset, cfg=cfg)

            # --- B. 提取绝对正确的 Ground Truth (直接从数据加载器提取) ---
            gt_x = traj['ego_pos'][:, 0]
            gt_y = traj['ego_pos'][:, 1]
            gt_vx = traj['ego_vel'][:, 0]
            gt_vy = traj['ego_vel'][:, 1]
            # 计算真实速度大小
            gt_v = np.sqrt(gt_vx**2 + gt_vy**2)

            # --- C. 运行模型推理 ---
            model_x, model_y, model_v = run_inference(model, env)

            # --- D. 绘图：仅在绘图阶段换算为 SI 单位 ---
            gt_x_m = gt_x * FT_TO_M
            gt_y_m = gt_y * FT_TO_M
            gt_v_mps = gt_v * FT_TO_M
            model_x_m = np.asarray(model_x) * FT_TO_M
            model_y_m = np.asarray(model_y) * FT_TO_M
            model_v_mps = np.asarray(model_v) * FT_TO_M

            fig, ax = plt.subplots(figsize=(5.1, 5.0))

            vmin_val, vmax_val = 0.0, 25.0
            y_min = min(np.min(gt_y_m), np.min(model_y_m)) - 6.0
            y_max = max(np.max(gt_y_m), np.max(model_y_m)) + 6.0
            road_x_min = cfg.X_MIN * FT_TO_M
            road_x_max = cfg.X_MAX * FT_TO_M
            lane_divider_x = (cfg.X_MIN + cfg.LANE_WIDTH) * FT_TO_M

            # Use the plot frame as the road edge; only draw the internal lane divider.
            ax.vlines(
                lane_divider_x,
                y_min,
                y_max,
                color="0.15",
                linestyle="--",
                linewidth=1.0,
                alpha=0.85,
                zorder=1,
            )

            # 绘制 Ground Truth 颜色渐变线
            points = np.array([gt_x_m, gt_y_m]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap='RdYlGn_r', linewidth=2.2, alpha=0.75, zorder=2)
            lc.set_array(gt_v_mps)
            lc.set_clim(vmin=vmin_val, vmax=vmax_val)
            ax.add_collection(lc)
            truth_line, = ax.plot(
                gt_x_m,
                gt_y_m,
                color="black",
                linewidth=1.2,
                alpha=0.65,
                label="Truth",
                zorder=3,
            )
            
            # 绘制 Model 轨迹散点
            pred_points = ax.scatter(
                model_x_m,
                model_y_m,
                c=model_v_mps,
                cmap='RdYlGn_r',
                s=13,
                edgecolors='black',
                linewidths=0.35,
                label='Prediction',
                zorder=4,
                vmin=vmin_val,
                vmax=vmax_val,
            )
            # 图片修饰
            ax.set_aspect(1.0 / 3.0)
            ax.set_anchor('W')
            ax.set_xlim(road_x_min, road_x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.tick_params(labelsize=9, top=False, right=False)

            cbar = plt.colorbar(pred_points, ax=ax, fraction=0.045, pad=0.12)
            cbar.ax.set_title('Speed (m/s)', pad=8, fontsize=10)
            cbar.ax.tick_params(labelsize=9, pad=4)

            ax.grid(False)

            right_xtick = max(t for t in ax.get_xticks() if road_x_min <= t <= road_x_max)
            ax.annotate(
                '(m)',
                xy=(right_xtick, 0.0),
                xycoords=ax.get_xaxis_transform(),
                xytext=(-1, -2),
                textcoords='offset points',
                ha='left',
                va='top',
                fontsize=10,
                annotation_clip=False,
            )
            ax.text(
                -0.18,
                0.965,
                '(m)',
                transform=ax.transAxes,
                ha='center',
                va='top',
                fontsize=10,
            )

            save_stem = os.path.splitext(filename)[0]
            save_path = os.path.join(output_dir, f"{i + 1:03d}_{save_stem}.png")
            fig.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.03)
            plt.close(fig)
            print(f"   -> 已保存: {save_path}")

        except Exception as e:
            print(f"[ERROR] 处理文件 {filename} 时发生错误: {str(e)}")
            traceback.print_exc()

    print("\n[OK] 所有批量绘图任务完成！")


if __name__ == "__main__":
    visualize_trajectory_batch()
