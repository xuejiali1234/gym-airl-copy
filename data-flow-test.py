import numpy as np
import torch
from configs.config import Config
from utils.data_loader import MergingDataset
from envs.merging_env import MergingEnv

def check_data_alignment():
    print("="*50)
    print("🚀 开始数据对齐检测 (Expert vs Environment)")
    print("="*50)
    
    # 1. 加载配置与数据集
    cfg = Config()
    print("[1] 正在加载专家数据集并计算全局统计量...")
    # 注意：这里的路径请确保与你的 config 中一致，或者手动传入你的数据路径
    data_paths = [
        "data/lane_change_trajectories-0750am-0805am",
        "data/lane_change_trajectories-0805am-0820am",
        "data/lane_change_trajectories-0820am-0835am"
    ]
    
    print("正在加载数据集，请稍候...")
    dataset = MergingDataset(data_path=data_paths)
    
    if len(dataset) == 0:
        print("❌ 数据集加载失败，请检查路径。")
        return

    # 2. 采样专家数据
    print("\n[2] 正在采样 100 步真实专家数据...")
    expert_batch = dataset.sample_expert_batch(batch_size=100)
    expert_state = expert_batch['state'].numpy()
    expert_goal = expert_batch['goal'].numpy()
    
    # 模拟 train_airl_baseline 中的拼接逻辑
    if getattr(cfg, 'ENABLE_GOAL_CONDITION', False):
        expert_obs = np.concatenate([expert_state, expert_goal], axis=-1)
    else:
        expert_obs = expert_state

    # 3. 采样环境数据
    print("\n[3] 初始化 Gym 环境并随机交互 100 步...")
    env = MergingEnv(dataset)
    
    # [修改点 1]: Gymnasium 的 reset() 返回 (obs, info)
    obs, _ = env.reset() 
    env_obs_list = [obs]
    
    for _ in range(100):
        # 随机乱打方向盘和油门
        action = env.action_space.sample() 
        
        # [修改点 2]: Gymnasium 的 step() 返回 5 个值
        obs, _, terminated, truncated, _ = env.step(action) 
        env_obs_list.append(obs)
        
        if terminated or truncated:
            obs, _ = env.reset()
            
    env_obs = np.array(env_obs_list)

    # 4. 对比与判定
    print("\n" + "="*50)
    print("📊 核心数据比对报告")
    print("="*50)
    
    # -- 维度比对 --
    print(f"👉 专家数据维度: {expert_obs.shape}")
    print(f"👉 环境数据维度: {env_obs.shape}")
    if expert_obs.shape[-1] == env_obs.shape[-1]:
        print("✅ [通过] 维度完美对齐！")
    else:
        print("❌ [失败] 维度不一致，请检查特征拼接！")

    # -- 数值量级比对 --
    print("\n👉 专家数据统计:")
    print(f"    Mean: {expert_obs.mean():.4f}, Std: {expert_obs.std():.4f}")
    print(f"    Min:  {expert_obs.min():.4f}, Max: {expert_obs.max():.4f}")
    
    print("\n👉 环境数据统计:")
    print(f"    Mean: {env_obs.mean():.4f}, Std: {env_obs.std():.4f}")
    print(f"    Min:  {env_obs.min():.4f}, Max: {env_obs.max():.4f}")

    print("\n" + "="*50)
    print("🔬 最终诊断结论")
    print("="*50)
    # 环境数据的平均值如果接近 0，且最大值不超过 50 (极少数异常点除外)，说明归一化成功
    if np.abs(env_obs.mean()) < 2.0 and np.abs(env_obs.max()) < 100.0:
        print("🎉 恭喜！数值量级对齐成功！")
        print("环境返回的数据已经完全在标准正态分布附近（消除几千的物理坐标）。")
        print("你可以放心敲下 `python train_airl_baseline.py` 开始训练了！")
    else:
        print("🚨 警告！数值量级依旧爆炸！")
        print("环境依然返回了巨大的物理原始值，请停止训练并检查 env._get_obs()。")

if __name__ == "__main__":
    check_data_alignment()