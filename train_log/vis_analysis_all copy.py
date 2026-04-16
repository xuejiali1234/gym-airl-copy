import pandas as pd
import matplotlib.pyplot as plt
import os

# 全局绘图样式设置
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

def get_column(df, possible_names):
    """辅助函数：兼容不同版本的 log 列名"""
    for name in possible_names:
        if name in df.columns:
            return name
    return None

def get_series(df, possible_names):
    """Return a numeric series for the first matching column, or None."""
    col = get_column(df, possible_names)
    if col is None:
        return None
    return pd.to_numeric(df[col], errors='coerce')

def plot_combined_figures(csv_path, save_dir):
    """一键读取数据，并将 7 张图绘制在同一张画布上 (3x3 布局)"""
    print(f"[*] 开始读取数据文件: {csv_path}")
    
    if not os.path.exists(csv_path):
        print(f"❌ 找不到数据文件: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    iterations = range(1, len(df) + 1)
    
    # 创建 3 行 3 列的子图画布
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    
    # ---------------------------------------------------------
    # 图一：Episode Reward (位置: [0, 0])
    # ---------------------------------------------------------
    rew_col = get_column(df, ['mean/gen/rollout/ep_rew_wrapped_mean', 'rollout/ep_rew_wrapped_mean'])
    if rew_col:
        axes[0, 0].plot(iterations, df[rew_col], color='#1f77b4', linewidth=2.0)
        axes[0, 0].set_title('Figure 1: Episode Reward', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Iterations', fontsize=12)
        axes[0, 0].set_ylabel('Reward', fontsize=12)
        axes[0, 0].grid(True, linestyle='--', alpha=0.6)

    # ---------------------------------------------------------
    # 图二：Discriminator Loss (位置: [0, 1])
    # ---------------------------------------------------------
    d_loss_col = get_column(df, ['mean/disc/disc_loss'])
    if d_loss_col:
        axes[0, 1].plot(iterations, df[d_loss_col], color='#d62728', linewidth=2.0)
        axes[0, 1].set_title('Figure 2: Discriminator Loss', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Iterations', fontsize=12)
        axes[0, 1].set_ylabel('Loss', fontsize=12)
        axes[0, 1].grid(True, linestyle='--', alpha=0.6)

    # ---------------------------------------------------------
    # 图三：Generator Value Loss (位置: [0, 2])
    # ---------------------------------------------------------
    v_loss_col = get_column(df, ['mean/gen/train/value_loss'])
    if v_loss_col:
        axes[0, 2].plot(iterations, df[v_loss_col], color='#2ca02c', linewidth=2.0)
        axes[0, 2].set_title('Figure 3: Generator Value Loss', fontsize=14, fontweight='bold')
        axes[0, 2].set_xlabel('Iterations', fontsize=12)
        axes[0, 2].set_ylabel('Value Loss', fontsize=12)
        axes[0, 2].grid(True, linestyle='--', alpha=0.6)

    # ---------------------------------------------------------
    # 图四：Policy Loss (位置: [1, 0]) - 按要求修改
    # ---------------------------------------------------------
    p_loss_col = get_column(df, ['mean/gen/train/policy_gradient_loss', 'mean/gen/train/loss'])
    if p_loss_col:
        axes[1, 0].plot(iterations, df[p_loss_col], color='#ff7f0e', linewidth=2.0)
        axes[1, 0].set_title('Figure 4: Policy Loss', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Iterations', fontsize=12)
        axes[1, 0].set_ylabel('Policy Loss', fontsize=12)
        axes[1, 0].grid(True, linestyle='--', alpha=0.6)

    # ---------------------------------------------------------
    # 图五：Discriminator vs Generator Accuracy (位置: [1, 1])
    # ---------------------------------------------------------
    acc_exp_col = get_column(df, ['mean/disc/disc_acc_expert'])
    acc_gen_col = get_column(df, ['mean/disc/disc_acc_gen'])
    if acc_exp_col and acc_gen_col:
        axes[1, 1].plot(iterations, df[acc_exp_col], label='Disc (Exp)', color='#1f77b4', linewidth=2.0)
        axes[1, 1].plot(iterations, df[acc_gen_col], label='Gen (Gen)', color='#d62728', linestyle='--', linewidth=2.0)
        axes[1, 1].axhline(y=0.5, color='gray', linestyle='-.', label='0.5 Line')
        axes[1, 1].set_title('Figure 5: Adversarial Accuracy', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Iterations', fontsize=12)
        axes[1, 1].set_ylabel('Accuracy', fontsize=12)
        axes[1, 1].set_ylim(0, 1.05)
        axes[1, 1].legend(loc='lower right', fontsize=10)
        axes[1, 1].grid(True, linestyle='--', alpha=0.6)

    # ---------------------------------------------------------
    # 图六：Action Standard Deviation (位置: [1, 2])
    # ---------------------------------------------------------
    std_col = get_column(df, ['mean/gen/train/std'])
    if std_col:
        axes[1, 2].plot(iterations, df[std_col], color='#8c564b', linewidth=2.0)
        axes[1, 2].set_title('Figure 6: Action Std Deviation', fontsize=14, fontweight='bold')
        axes[1, 2].set_xlabel('Iterations', fontsize=12)
        axes[1, 2].set_ylabel('Action Std', fontsize=12)
        axes[1, 2].grid(True, linestyle='--', alpha=0.6)

    # ---------------------------------------------------------
    # 图七：Approx KL Divergence (位置: [2, 0]) - 按要求移至此处
    # ---------------------------------------------------------
    kl_col = get_column(df, ['mean/gen/train/approx_kl'])
    if kl_col:
        axes[2, 0].plot(iterations, df[kl_col], color='#9467bd', linewidth=2.0)
        axes[2, 0].set_title('Figure 7: Approx KL Divergence', fontsize=14, fontweight='bold')
        axes[2, 0].set_xlabel('Iterations', fontsize=12)
        axes[2, 0].set_ylabel('KL Divergence', fontsize=12)
        axes[2, 0].grid(True, linestyle='--', alpha=0.6)

    # 隐藏空余的子图 (最后两个格子)
    safety_aux = get_series(df, ['mean/disc/disc_safety_aux', 'disc_safety_aux'])
    if safety_aux is not None:
        axes[2, 1].plot(iterations, safety_aux, color='#17becf', linewidth=2.0)
        axes[2, 1].set_title('Figure 8: Safety Auxiliary Loss', fontsize=14, fontweight='bold')
        axes[2, 1].set_xlabel('Iterations', fontsize=12)
        axes[2, 1].set_ylabel('Safety Aux Loss', fontsize=12)
        axes[2, 1].grid(True, linestyle='--', alpha=0.6)
    else:
        axes[2, 1].axis('off')

    safety_reg = get_series(df, ['mean/disc/disc_safety_reg_loss', 'disc_safety_reg_loss'])
    if safety_reg is None:
        safety_weight = get_series(df, ['mean/disc/disc_safety_weight', 'disc_safety_weight'])
        if safety_aux is not None and safety_weight is not None:
            safety_reg = safety_aux * safety_weight

    if safety_reg is not None:
        axes[2, 2].plot(iterations, safety_reg, color='#bcbd22', linewidth=2.0)
        axes[2, 2].set_title('Figure 9: Safety Reg Loss', fontsize=14, fontweight='bold')
        axes[2, 2].set_xlabel('Iterations', fontsize=12)
        axes[2, 2].set_ylabel('Reg Loss', fontsize=12)
        axes[2, 2].grid(True, linestyle='--', alpha=0.6)
    else:
        axes[2, 2].axis('off')

    plt.tight_layout(pad=3.0)
    save_path = os.path.join(save_dir, 'Full_Training_Analysis_9Figs.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved 9-figure analysis dashboard to: {save_path}")
    plt.show()
    return
    print(f"🎉 7张图像绘制完毕！已合成大图保存至: {save_path}")
    plt.show()

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_absolute_path = os.path.join(script_dir, 'baseline_attn_goal_safe_branch_aux_rnorm_20260416_105052', 'progress.csv')
    plot_combined_figures(csv_absolute_path, script_dir)
