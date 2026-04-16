import pandas as pd
import matplotlib
matplotlib.use('Agg') # 配置非交互式后端
import matplotlib.pyplot as plt

# Load the new CSV
try:
    df = pd.read_csv(r'train_log\train_20260327_120336.csv')
    print("Statistics:")
    print(df.describe())

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Return
    if 'Return' in df.columns:
        axes[0, 0].plot(df['Epoch'], df['Return'], label='Return', color='green', linewidth=2)
    
    axes[0, 0].set_title('Average Return (Higher is Better)')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    # Add a zero line to show where it becomes positive
    axes[0, 0].axhline(y=0, color='black', linestyle='--')

    # 2. Discriminator Loss
    axes[0, 1].plot(df['Epoch'], df['D_Loss'], color='red', linewidth=2)
    axes[0, 1].axhline(y=1.386, color='blue', linestyle='--', label='Nash Equilibrium (1.386)')
    axes[0, 1].set_title('Discriminator Loss (Target ~1.38)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # 3. Policy Loss
    axes[1, 0].plot(df['Epoch'], df['P_Loss'], color='purple')
    axes[1, 0].set_title('Policy Loss (Stability)')
    axes[1, 0].grid(True)

    # 4. Collision Rate
    if 'Collision_Rate' in df.columns:
        axes[1, 1].plot(df['Epoch'], df['Collision_Rate'], color='orange', linewidth=2)
        axes[1, 1].set_title('Collision Rate (Lower is Better)')
        axes[1, 1].set_ylabel('Rate')
        axes[1, 1].set_ylim(0, 1.05)
    elif 'Mean_Beta' in df.columns:
        axes[1, 1].plot(df['Epoch'], df['Mean_Beta'], color='orange')
        axes[1, 1].set_title('Mean Beta (Confidence)')
        axes[1, 1].set_ylim(0, 1.1)
    
    axes[1, 1].grid(True)
    plt.tight_layout()
    
    # 【修改 1】：指定相对路径，让它保存进 train_log 文件夹
    plt.savefig(r'train_log\analysis_training_stats.png')

except Exception as e:
    print(f"Error: {e}")