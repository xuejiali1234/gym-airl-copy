import os
import pandas as pd
import matplotlib
matplotlib.use('Agg') # 配置非交互式后端
import matplotlib.pyplot as plt

# 1. 设置最新的 CSV 路径
csv_filename = 'train_20260402_201321.csv'
csv_path = os.path.join('train_log', csv_filename)
if not os.path.exists(csv_path):
    csv_path = csv_filename # 备用路径

try:
    df = pd.read_csv(csv_path)
    print(f"✅ 成功加载数据: {csv_path} (共 {len(df)} 轮)")

    # 设置画布格式：4行2列，尺寸更大以容纳所有信息
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    fig.suptitle('GC-AIRL Comprehensive Training Analysis', fontsize=20, fontweight='bold', y=0.98)

    # 为了防止某些旧版CSV缺少列报错，封装一个安全绘图函数
    def safe_plot(ax, x_col, y_cols, title, ylabel, colors, labels=None, hlines=None):
        plotted = False
        for i, y_col in enumerate(y_cols):
            if y_col in df.columns:
                label = labels[i] if labels else y_col
                ax.plot(df[x_col], df[y_col], label=label, color=colors[i], linewidth=2)
                plotted = True
        
        if plotted:
            ax.set_title(title, fontsize=14)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            if hlines:
                for y_val, c, ls, lbl in hlines:
                    ax.axhline(y=y_val, color=c, linestyle=ls, label=lbl, alpha=0.8)
            
            ax.legend(loc='best')
        else:
            ax.text(0.5, 0.5, 'Data Not Found', ha='center', va='center', fontsize=15, color='gray')

    # --- [第一排] 物理环境表现 (Environment Metrics) ---
    # 1. Total Return
    safe_plot(axes[0, 0], 'Epoch', ['Return'], '1. Average Return (Higher is Better)', 'Return', ['green'], 
              hlines=[(0, 'black', '--', 'Zero Baseline')])
    
    # 2. Success Rate vs Collision Rate
    safe_plot(axes[0, 1], 'Epoch', ['Success_Rate', 'Collision_Rate'], '2. Success vs Collision Rate', 'Rate (0~1)', 
              ['royalblue', 'darkorange'], labels=['Success Rate', 'Collision Rate'])

    # --- [第二排] AIRL 博弈指标 (Adversarial Metrics) ---
    # 3. Discriminator Loss
    safe_plot(axes[1, 0], 'Epoch', ['D_Loss'], '3. Discriminator Loss', 'Loss', ['red'],
              hlines=[(1.386, 'blue', '--', 'Nash Eq (~1.386)')])
    
    # 4. Discriminator Output Scores (Core AIRL Metric!)
    # 这是判断模型是否收敛的最核心图表：专家得分与策略得分的拉锯战
    safe_plot(axes[1, 1], 'Epoch', ['Exp_Score', 'Pol_Score'], '4. D-Net Output Scores (Nash Eq Target)', 'Probability', 
              ['darkgreen', 'purple'], labels=['Expert Score', 'Policy Score'])

    # --- [第三排] PPO 策略与价值损失 (PPO Metrics) ---
    # 5. Policy Loss
    safe_plot(axes[2, 0], 'Epoch', ['P_Loss'], '5. Policy Loss (P_Loss)', 'Loss', ['purple'])
    
    # 6. Value Loss
    safe_plot(axes[2, 1], 'Epoch', ['V_Loss'], '6. Value Loss (V_Loss)', 'Loss', ['teal'])

   # --- [第四排] 其他辅助指标 ---
    # 7. Safety Regulator Loss (专门约束判别器的隐式惩罚)
    if 'Reg_Loss' in df.columns:
        safe_plot(axes[3, 0], 'Epoch', ['Reg_Loss'], '7. Safety Regulator Loss (Reg_Loss)', 'Loss', ['brown'])
    else:
        axes[3, 0].text(0.5, 0.5, 'Reg_Loss Not Logged', ha='center', va='center', fontsize=12, color='gray')
        axes[3, 0].set_title('7. Safety Regulator Loss')
        axes[3, 0].axis('off')
    
    # 8. Safety Network Loss (安全模块的二分类交叉熵损失)
    # 请确保你的 CSV 文件中包含了 'S_Loss' 这一列。如果是其他名字 (如 'Safety_Loss')，请自行修改。
    if 'S_Loss' in df.columns:
        safe_plot(axes[3, 1], 'Epoch', ['S_Loss'], '8. Safety Network Loss (S_Loss)', 'BCE Loss', ['darkred'])
    elif 'Safety_Loss' in df.columns: # 兼容不同命名
        safe_plot(axes[3, 1], 'Epoch', ['Safety_Loss'], '8. Safety Network Loss', 'BCE Loss', ['darkred'])
    else:
        axes[3, 1].text(0.5, 0.5, 'Safety Module Disabled\\nor Loss Not Logged', ha='center', va='center', fontsize=12, color='gray')
        axes[3, 1].set_title('8. Safety Network Loss')
        axes[3, 1].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # 留出顶部大标题的空间
    
    # 2. 动态保存路径
    save_dir = 'train_log' if os.path.exists('train_log') else '.'
    save_path = os.path.join(save_dir, 'analysis_all.png')
    
    plt.savefig(save_path, dpi=150) # 提高分辨率
    print(f"✅ 综合全景分析图已保存至:\n 👉 {os.path.abspath(save_path)}")

except Exception as e:
    print(f"❌ 运行出错: {e}")