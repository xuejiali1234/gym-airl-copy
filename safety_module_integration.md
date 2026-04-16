可以，已经先按你现在这套 **Gymnasium + SB3 + imitation AIRL** 工程，给你做了一版“**安全 Q 融合判别器 + 轻量安全损失干预判别器**”的可接入实现。

文件我整理成一套干净版，直接放在 `model/` 下用：

[safety_q_module.py](sandbox:/mnt/data/safety_qdisc_ready/model/safety_q_module.py)
[safety_oracle_q.py](sandbox:/mnt/data/safety_qdisc_ready/model/safety_oracle_q.py)
[safety_pretrain_q.py](sandbox:/mnt/data/safety_qdisc_ready/model/safety_pretrain_q.py)
[safety_airl.py](sandbox:/mnt/data/safety_qdisc_ready/model/safety_airl.py)
[接入说明 integration.md](sandbox:/mnt/data/safety_qdisc_ready/integration.md)

先说判断。

你**当前安全模块方向是对的，但还没完全到你图里的形态**。

当前合理的地方有三点。

第一，你现在已经把安全分支作为**判别器/奖励网络的一个并行分支去做特征融合**，不是继续把安全奖励硬堆回 `env.step()`，这条路线是对的。你现在的 `SafeAttentionRewardNet` 已经是“行为分支 + 安全分支 -> 拼接 -> head”的结构，这和你图里的“安全网络 Q 融入原始判别器”在大方向上是一致的。

第二，你把安全网先预训练、再冻结后接到 reward/discriminator 里，这也是合理的。这样能避免对抗训练早期把安全网络一起拖乱。你当前代码就是先 `pretrain_safety_network(...)`，再把 `safety_net` 传给 `SafeAttentionRewardNet(..., freeze_safety=True)`。

第三，你的数据表示本身支持做 `Q_safe(s,a)`。你现在的任务定义本来就是 **16 维状态 + 2 维动作**，动作还是按物理极限做了固定比例归一化；这和论文里“状态是 16 维、动作是二维加速度”的 formulation 是一致的。论文的 discriminator 也是按 `c_ω(s,a,g)` 这种 state-action-goal 形式写的。 

但当前不合理、或者说还没做完的地方，也很明确。

最核心的问题是：**你现在的安全网还不是 `Q_safe(s,a)`，而更像 `risk(s)` / `V_safe(s)`。**
因为你当前训练脚本里实例化的是：

```python
safety_net = SafetyNetwork(state_dim=16, hidden_dim=128, use_action=False)
```

也就是说，安全网现在根本没吃动作。这样它学到的不是你图里的 `(S,A) -> S-Net -> Q_safe(s,a)`，而只是“当前状态危险不危险”。

第二个问题是：**你现在的 `SAFETY_REGULATOR_COEFF` 在当前 Gym/imitation 路径里并没有真正参与判别器 loss。**
因为你现在训练器还是直接用 `imitation.algorithms.adversarial.airl.AIRL`，而 `imitation` 默认的 `train_disc()` 就是标准的 BCE-with-logits 判别器更新；AIRL 的 discriminator logit 也是 `reward_net(state, action, next_state, done) - log π(a|s)`。默认实现里没有你旧版 `gc_airl.py` 那种额外的 safety regulator 项。  ([Imitation][1])

第三个问题是：**你旧版 non-gym 代码里其实已经有你真正想要的那个“轻微干预判别器”的思路，但这部分还没有迁移到 imitation AIRL。**
你旧版 `gc_airl.py` 里是：

* 先算 AIRL 主 loss
* 再算 `regulator_loss = pol_risk * clamp(pol_g_score, ...)`
* 最后 `total_loss = airl_loss + SAFETY_REGULATOR_COEFF * regulator_loss`

这个思路本身是对的，而且比“直接把安全惩罚加到 policy reward”更符合你现在说的“**轻微干预判别器**”。

第四个小一致性问题也要提一句：
你现在 `env.step()` 里虽然会算 `is_safety_success`，但 `info` 里只回传了 `is_success` 和 `is_collided`；可训练脚本又在 `Monitor(..., info_keywords=(..., "is_safety_success"))` 里盯这个字段。这个评估口径最好顺手补齐。 

---

## 我这次给你实现的是什么

这版实现，做了三件关键事。

### 1. 把安全网升级成真正的 `Q_safe(s,a)`

新的 `SafetyQNetwork` 是 action-aware 的：

* 输入：前 16 维核心状态 + 2 维动作
* 输出：安全 logit / risk，以及一个 safety latent feature
* 这样就能和你图里 `(S,A) -> Trained S-Net -> Q_safe(s,a)` 对上

对应文件是：

[safety_q_module.py](sandbox:/mnt/data/safety_qdisc_ready/model/safety_q_module.py)

---

### 2. 把安全分支按你图里的方式融合进判别器

新的 `SafeQAttentionRewardNet` 结构就是：

* 行为分支：`[state, attention(state), action] -> MLP`
* 安全分支：`S-Net(state, action) -> [q_safe, safety_feature] -> MLP`
* 拼接后再过 `head -> f_theta(s,a)`

也就是说，**不仅判动作像不像专家，也看这个 state-action 是否安全**。

这比你当前版更贴近图里的结构，因为现在安全分支是真正的 `(s,a)` 分支，不是 state-only 分支。

---

### 3. 给判别器加一个“轻量安全损失”

我新增了 `MildSafetyAIRL(AIRL)`，只改 discriminator update，不改 generator 主训练逻辑。

它做的是：

[
L_{disc} = L_{BCE} + \lambda_s \cdot L_{aux}
]

其中

[
L_{aux} = \mathbb{E}*{(s,a)\sim gen}\big[q*{safe}(s,a)\cdot \text{softplus}(\text{disc_logit}(s,a))\big]
]

直观意思就是：

* 只看 **generator 样本**
* 如果一个样本 **风险高**
* 而判别器还把它打得很 **expert-like**
* 那就轻轻罚一下

这就正好对应你说的“**安全损失轻微干预判别器**”。

不是推翻 AIRL 主目标，只是轻推它。

---

## 你现在怎么接

你当前 `train_airl_baseline.py` 里，最小改法就是四处。

### 先替换导入

```python
from model.safety_q_module import SafetyQNetwork, SafeQAttentionRewardNet, SafeQMLPRewardNet
from model.safety_oracle_q import SafetyOracleQ
from model.safety_pretrain_q import pretrain_safety_q_network
from model.safety_airl import MildSafetyAIRL
```

### 再把安全网改成 action-aware 预训练

把你现在的：

```python
safety_oracle = SafetyOracle(cfg, train_dataset.expert_mean, train_dataset.expert_std)
safety_net = SafetyNetwork(state_dim=16, hidden_dim=128, use_action=False)
pretrain_stats = pretrain_safety_network(...)
```

改成：

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

### 再把 reward net 换掉

Attention 版改成：

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

### 最后把 AIRL trainer 换成带轻量安全损失的版本

把：

```python
airl_trainer = AIRL(...)
```

改成：

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

---

## 我对当前安全模块的最终评价

一句话概括：

> **你当前的安全模块“方向合理，但还差最后两步”。**

合理在于：

* 安全信息已经走进了判别器路径，而不是继续堆手工环境奖励
* 先预训练再冻结的工程思路是对的
* 你的 state/action 定义本身支持做 `Q_safe(s,a)`   

不合理在于：

* 现在还是 `use_action=False`，所以不是 `Q_safe(s,a)` 
* 现在还是 vanilla imitation AIRL，所以 `SAFETY_REGULATOR_COEFF` 没真正进 discriminator loss   ([Imitation][1])
* 你旧版 `gc_airl.py` 里那套 regulator 逻辑还没迁到当前管线里

最合理的调整顺序就是：

1. 先把安全网改成 **action-aware**
2. 再把 safety oracle / pretrain 改成 **state-action supervision**
3. 再用一个很小的 `λ_s` 去**轻量干预判别器**
4. 先不要把安全惩罚直接塞回 generator reward

---

## 我建议你起步的安全超参

第一轮不要太猛，先这样：

```python
SAFETY_REGULATOR_COEFF = 0.02
```

不是 0.1，也不是 0.2。
因为你现在要的是“轻微干预”，不是让安全分支把 AIRL 主目标压掉。

同时保持：

* `freeze_safety=True`
* `use_action=True`
* `goal_bonus` 先别动，先保持你当前较稳那档

---

你要是愿意，我下一步直接给你一版**基于你当前 `train_airl_baseline(11).py` 的逐行 patch**，把导入、预训练、reward_net、AIRL trainer 四处一次改完。

[1]: https://imitation.readthedocs.io/en/latest/_modules/imitation/algorithms/adversarial/common.html "imitation.algorithms.adversarial.common - imitation"
