# 安全 Q 融合判别器 + 轻量判别器安全损失 接入说明

## 1. 新增文件

- `model/safety_q_module.py`
- `model/safety_oracle_q.py`
- `model/safety_pretrain_q.py`
- `model/safety_airl.py`

## 2. 训练脚本中替换导入

```python
from model.safety_q_module import SafetyQNetwork, SafeQAttentionRewardNet, SafeQMLPRewardNet
from model.safety_oracle_q import SafetyOracleQ
from model.safety_pretrain_q import pretrain_safety_q_network
from model.safety_airl import MildSafetyAIRL
```

## 3. 预训练 S-Net（改成 state-action）

```python
safety_oracle = SafetyOracleQ(cfg, train_dataset.expert_mean, train_dataset.expert_std)
safety_net = SafetyQNetwork(
    state_dim=16,
    action_dim=2,
    hidden_dim=128,
    use_action=True,
)
pretrain_stats = pretrain_safety_q_network(
    safety_net,
    train_dataset,
    safety_oracle,
    device=device,
    epochs=15,
    batch_size=512,
    lr=cfg.SAFETY_LEARNING_RATE,
    synthetic_multiplier=1.0,
    verbose=True,
)
```

## 4. 替换 reward net

Attention 版：

```python
base_reward_net = SafeQAttentionRewardNet(
    observation_space=env.observation_space,
    action_space=env.action_space,
    safety_net=safety_net,
    hidden_dim=64,
    safety_embed_dim=32,
    freeze_safety=True,
    fuse_safety_feature=True,
)
reward_net = GoalRewardWrapper(
    base_reward_net,
    expert_mean_x=train_dataset.expert_mean[0],
    expert_std_x=train_dataset.expert_std[0],
    divider_x=divider_x,
    goal_bonus=0.5,
)
```

MLP 版：

```python
base_reward_net = SafeQMLPRewardNet(
    observation_space=env.observation_space,
    action_space=env.action_space,
    safety_net=safety_net,
    safety_embed_dim=32,
    freeze_safety=True,
    fuse_safety_feature=True,
)
```

## 5. 替换 AIRL trainer

```python
airl_trainer = MildSafetyAIRL(
    demonstrations=expert_trajectories,
    demo_batch_size=256,
    gen_replay_buffer_capacity=2048,
    n_disc_updates_per_round=4,
    venv=env,
    gen_algo=learner,
    reward_net=reward_net,
    allow_variable_horizon=True,
    custom_logger=custom_logger,
    safety_loss_weight=cfg.SAFETY_REGULATOR_COEFF,
    disc_opt_kwargs=dict(
        lr=cfg.DISCRIMINATOR_LEARNING_RATE,
        weight_decay=1e-3,
    ),
)
```

## 6. 推荐新增配置

```python
SAFETY_REGULATOR_COEFF = 0.02   # 起步建议 0.02~0.05
SAFETY_USE_ACTION = True
SAFETY_FUSE_FEATURE = True
```

## 7. 一句话原则

- 先让 `Q_safe(s,a)` 真的成立；
- 再让它作为特征融入判别器；
- 最后只用一个很小的 auxiliary loss 轻推判别器，避免把 AIRL 主目标改坏。
