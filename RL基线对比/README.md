# PPO-RL Baseline

这套代码是独立于 AIRL 主线的非 AIRL PPO-RL 手工奖励基线。

目标：
- 复用当前 `MergingEnv / MergingDataset / PPO / 评估口径`
- 只替换训练奖励
- 不改动 `train_airl_baseline.py` 和主配置默认值

运行前请先进入项目使用的 `gym_airl` Python 环境；`run_ppo_rl_baseline_sweep.py` 会继承你当前启动它的解释器。

## 文件

- `ppo_rl_reward_wrapper.py`
  - 手工 reward wrapper
- `train_ppo_rl_baseline.py`
  - 单次 PPO-RL 训练入口
- `run_ppo_rl_baseline_sweep.py`
  - 5 组 baseline sweep
- `evaluate_ppo_rl_baseline.py`
  - fixed protocol 复评入口

## 默认主版本

- `PPO_RL_Goal_Default`
- `goal on`
- `attention off`
- `safety off`
- reward = efficiency + safety(TTC) + THW + comfort + goal progress + terminal

## 常用命令

单次训练：

```powershell
python .\RL基线对比\train_ppo_rl_baseline.py --tag PPO_RL_Goal_Default
```

Sweep 预览：

```powershell
python .\RL基线对比\run_ppo_rl_baseline_sweep.py --dry-run
```

3 epoch 烟测：

```powershell
python .\RL基线对比\run_ppo_rl_baseline_sweep.py --only PPO_RL_Goal_Default --epochs 3
```

固定协议复评：

```powershell
python .\RL基线对比\evaluate_ppo_rl_baseline.py --checkpoint PATH_TO_ZIP --tag PPO_RL_Goal_Default
```

如果仓库里没有现成的 `common15_trajectory_summary_for_gpt.csv`，复评脚本会自动退回只跑 full split。

## 日志

训练输出仍保存在项目根目录的 `train_log/` 下，便于与 P30 直接横向对比。
