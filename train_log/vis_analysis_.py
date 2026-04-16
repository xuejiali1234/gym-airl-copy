import os
import pandas as pd
import matplotlib
matplotlib.use('Agg') # 配置非交互式后端 (只保存图片，不弹窗)
import matplotlib.pyplot as plt

# 1. 动态获取正确的 CSV 路径
csv_path = r'train_log\train_20260326_173324.csv'
if not os.path.exists(csv_path):
    csv_path = 'train_20260326_173324.csv' # 备用路径，防止你在 train_log 目录下直接运行

try:
    df = pd.read_csv(csv_path)
    print(f"✅ 数据加载成功！共读取到 {len(df)} 轮 (Epoch) 的数据。")

    # 创建 2x2 画布
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # [图 1] 绘制 Return (总回报)
    if 'Return' in df.columns:
        axes[0, 0].plot(df['Epoch'], df['Return'], label='Return', color='green', linewidth=2)
    axes[0, 0].set_title('Average Return (Higher is Better)')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    axes[0, 0].axhline(y=0, color='black', linestyle='--')

    # [图 2] 绘制 D_Loss (判别器损失)
    if 'D_Loss' in df.columns:
        axes[0, 1].plot(df['Epoch'], df['D_Loss'], color='red', linewidth=2)
        axes[0, 1].axhline(y=1.386, color='blue', linestyle='--', label='Nash Eq (1.386)')
    axes[0, 1].set_title('Discriminator Loss (Target ~1.38)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # [图 3] 绘制 P_Loss (策略损失)
    if 'P_Loss' in df.columns:
        axes[1, 0].plot(df['Epoch'], df['P_Loss'], color='purple', linewidth=2)
    axes[1, 0].set_title('Policy Loss (Watch for Spikes!)')
    axes[1, 0].grid(True)

    # [图 4] 绘制 Collision Rate (碰撞率)
    if 'Collision_Rate' in df.columns:
        axes[1, 1].plot(df['Epoch'], df['Collision_Rate'], color='orange', linewidth=2)
        axes[1, 1].set_title('Collision Rate (Lower is Better)')
        axes[1, 1].set_ylabel('Rate')
        axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].grid(True)

    plt.tight_layout()
    
    # 2. 明确保存路径
    save_path = r'train_log\analysis_training_stats.png'
    # 如果你在没有 train_log 的地方运行，就保存在当前目录
    if not os.path.exists('train_log'):
        save_path = 'analysis_training_stats.png'
        
    plt.savefig(save_path)
    print(f"✅ 图片已成功保存至绝对路径:\n 👉 {os.path.abspath(save_path)}")
    
    # 删除了 plt.show()，彻底消灭红色 Warning 警告

except FileNotFoundError:
    print(f"❌ 找不到文件！请检查路径 {csv_path} 是否存在 CSV 文件。")
except Exception as e:
    print(f"❌ 运行出错: {e}")