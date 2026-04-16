import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

# ==========================================
# 1. 配置参数
# ==========================================
DT = 0.1  # 环境的时间步长
# 数据集目录 (相对于当前脚本的路径)
DATA_DIRS = [
    'data/lane_change_trajectories-0750am-0805am',
    'data/lane_change_trajectories-0805am-0820am',
    'data/lane_change_trajectories-0820am-0835am'
]

def calculate_percentiles():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    
    all_acc_y = [] # 纵向加速度 (纵向控制)
    all_acc_x = [] # 横向加速度 (对应转向控制)

    # 1. 搜集所有 CSV 文件
    csv_files = []
    for d in DATA_DIRS:
        search_path = os.path.join(curr_dir, d, '**', '*.csv')
        csv_files.extend(glob.glob(search_path, recursive=True))

    if not csv_files:
        print("❌ 找不到任何 CSV 文件，请检查 DATA_DIRS 路径是否正确！")
        return

    print(f"🔍 共找到 {len(csv_files)} 个轨迹文件，开始全量扫描...")

    # 2. 遍历提取物理量
    for f in tqdm(csv_files, desc="Processing Trajectories"):
        try:
            df = pd.read_csv(f)
            
            # 提取速度
            if 'KF_Vel_Y' not in df.columns or 'KF_Vel_X' not in df.columns:
                continue
                
            vy = df['KF_Vel_Y'].values
            vx = df['KF_Vel_X'].values
            
            # 通过速度差分计算加速度 (a = dv / dt)
            # 使用 prepend=vy[0] 使得长度与原数组一致，且第一帧加速度记为0
            ay = np.diff(vy, prepend=vy[0]) / DT
            ax = np.diff(vx, prepend=vx[0]) / DT
            
            # 强化学习边界我们只关心幅值 (绝对值)
            all_acc_y.extend(np.abs(ay))
            all_acc_x.extend(np.abs(ax))
            
        except Exception as e:
            # 忽略损坏或格式错误的文件
            continue

    if not all_acc_y:
        print("❌ 未能提取到任何有效数据。")
        return

    # 转为 Numpy 数组进行高速统计
    all_acc_y = np.array(all_acc_y)
    all_acc_x = np.array(all_acc_x)

    # 3. 统计与打印报告
    print("\n" + "="*50)
    print(f"📊 扫描结果分析报告 (总数据点: {len(all_acc_y):,})")
    print("="*50)
    
    # 观察不同分位数的跨度，识别噪声
    quantiles = [90, 95, 99, 99.5, 99.9, 99.99, 100]
    
    print(f"{'分位数':<10} | {'纵向加速度 (Acc_Y)':<20} | {'横向加速度 (Acc_X)':<20}")
    print("-" * 55)
    
    for q in quantiles:
        py = np.percentile(all_acc_y, q)
        px = np.percentile(all_acc_x, q)
        
        label = f"{q}%" if q != 100 else "Max (100%)"
        print(f"{label:<12} | {py:>10.2f} ft/s^2      | {px:>10.2f} ft/s^2")

    print("\n" + "="*50)
    print("💡 Config.py 终极修改建议")
    print("="*50)
    p999_y = np.percentile(all_acc_y, 99.9)
    p999_x = np.percentile(all_acc_x, 99.9)
    
    print("请忽略 100% 的极大值 (通常为碰撞或雷达噪声)。")
    print("建议采用 99.9% 分位数作为物理边界：")
    print(f"PHYS_ACC_MAX = {p999_y:.1f}")
    print(f"PHYS_STEER_MAX = {p999_x:.1f}")
    print("="*50)

if __name__ == "__main__":
    calculate_percentiles()