# A-Train-2 及后续 A-gate 实验总结

## 1. 背景与目标

在 `U220_D230_epoch290 + v10b_leadgap_policy_veto_recovery` 这条主线上，我们已经得到了一个比较稳的部署方案：

- `full217`
  - `merge_success_rate = 0.8618`
  - `endpoint_success_rate = 0.9124`
  - `safety_success_rate = 0.9124`
  - `collision_rate = 0.0000`
  - `shield_intervention_rate = 0.3332`
- `hard15`
  - `merge_success_rate = 0.8000`
  - `endpoint_success_rate = 1.0000`
  - `safety_success_rate = 1.0000`
  - `collision_rate = 0.0000`
  - `shield_intervention_rate = 0.6023`

对应评估目录：

- `train_log/safety_shield_eval_v10b_leadgap_policy_veto_recovery_full217_20260428`

这条主线的问题不是“安不安全”，而是：**shield 依赖仍然偏高**。  
因此后续 `A-gate` 系列实验的目标，不是替换 `v10b` 动作层，而是尝试训练一个 **risk gate**：

- 让它在高风险状态下调用 `v10b`
- 在明显安全的状态下放行原策略
- 在不破坏 `collision=0` 和主协议指标的前提下，把 `shield_rate` 压低


## 2. A-Train-2 之前的起点

在最早的 A 方案中，我们已经做过一版 4 头 risk classifier：

- `intervene`
- `critical`
- `risk_side`
- `risk_reason`

随后做了 `A-Calib-1` 阈值校准，发现一个非常明显的现象：

- 只要 calibration 目标继续强烈偏向 `false_negative=0`
- 最终最优解就会自动落在“几乎每步都开 gate”的区域

旧版 A 的 strict best 大致表现为：

- `full217 collision = 0`
- `full217 merge / endpoint / safety` 都能守住
- 但 `full217 shield_rate ≈ 0.9642`

这说明旧版 A 能当“高召回风险检测器”，但**完全不能当实用 gate**。


## 3. A-Train-2：重训 binary/critical classifier

### 3.1 设计目的

`A-Train-2` 的核心思路是：

- 保留 4 头结构
- 但把 `intervene` 和 `critical` 变成主任务
- `risk_side / risk_reason` 只保留为弱辅助头
- 继续只做 **pre-gate**，不碰 actor，不碰 AIRL 主训练

### 3.2 输入与输出

输入保持不变：

- `obs`
- `policy_action`
- `lead_gap`
- `lead_thw`
- `follow_gap`
- `follow_thw`

输出仍为 4 个 head：

- `intervene`
- `critical`
- `risk_side`
- `risk_reason`

### 3.3 标签与训练目标

`A-Train-2` 的关键改动是标签和损失：

- `intervene`
  - `shield_intervened=True -> 1.0`
  - `warning=True and not intervened -> 0.4`
  - `normal_safe -> 0.0`
- `critical`
  - `critical_risk=True -> 1.0`
  - 其余 `0.0`

损失权重：

- `3.0 * weighted_bce(intervene)`
- `6.0 * weighted_bce(critical)`
- `0.1 * weighted_ce(risk_side)`
- `0.1 * weighted_ce(risk_reason)`

样本权重：

- `hard15 multiplier = 5.0`
- `intervened = 3.0`
- `warning-only = 1.5`
- `normal-safe = 1.0`
- `critical positive = 10.0`

### 3.4 A-Train-2 训练产物

训练目录：

- `train_log/v10b_risk_classifier_a2_20260429_165045`

保存了：

- `epoch_0.pt`
- `epoch_5.pt`
- `epoch_10.pt`
- `epoch_15.pt`
- `epoch_20.pt`
- `best_checkpoint.pt`


## 4. A-Train-2 训后第一次 calibration 结果

### 4.1 strict calibration 结论

在 `A-Train-2` 训练完成后，我们对 `epoch0/5/10/15/20` 做了完整阈值扫：

- `tau_intervene ∈ {0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50}`
- `tau_critical ∈ {0.05, 0.10, 0.20, 0.30}`

strict 规则下，A2 的 best 依然是“高召回、强开门”：

- `collision = 0`
- `merge / endpoint / safety` 可守住
- 但 `shield_rate` 仍接近全开，约 `0.9615`

这说明：

1. A2 没有把 gate 真正收紧
2. 它仍然主要在追求“不要漏掉任何可能危险状态”
3. 因此部署上不具备替代 baseline 的价值


## 5. 为什么 A2 会把 gate 撑得很大

后续我们围绕三个问题做了专门分析：

1. 为什么 `tau_critical=0.05` 会把 gate 撑开
2. `raw_v10b_critical_rule` 是否过于保守
3. 是否要把 `critical override` 改成只对部分风险类型生效

### 5.1 结论一：`tau_critical=0.05` 会明显放大 gate

原因不是它单独有问题，而是三件事叠在一起：

- `call_shield = raw_critical OR p_critical >= tau_critical OR p_intervene >= tau_intervene`
- `critical` 头训练权重很高
- calibration strict 规则强烈偏向 `false_negative = 0`

所以低阈值下，最容易被选中的就是：

- 安全性很好
- 但 shield 调用极多

### 5.2 结论二：`raw_v10b_critical_rule` 作为外层 hard override 偏保守

通过分析 teacher v2 数据集可以看到：

- `full` 中 `critical_risk=True` 占比约 `24.0%`
- `hard15` 中 `critical_risk=True` 占比约 `68.6%`

更重要的是，在 `full` 里还存在大量：

- `critical=True`
- 但 `shield_intervened=False`

这说明 `raw critical` 更像“值得再看一眼”的预警信号，  
而不是“必须无条件进 shield”的 hard override。


## 6. 针对 raw critical 的 focused calibration 实验

为了验证 `raw critical` 到底是不是主矛盾，我们没有继续重训，而是做了三组 focused calibration。

### 6.1 实验一：保留原始 `raw critical`

对应 practical best：

- 目录：`train_log/v10b_risk_calib_a2_20260429_165045/practical_best_config.json`
- `checkpoint = epoch_10`
- `tau_intervene = 0.5`
- `tau_critical = 0.3`
- `full217 shield_rate = 0.7089`
- `full217 false_negative_rate = 0.0673`
- `hard15 shield_rate = 0.9464`

这是 A2 线上最实用的一版。

### 6.2 实验二：`raw critical override` 只对 `overlap_or_ttc` 生效

对应目录：

- `train_log/v10b_risk_calib_a2_epoch15_overlap_ttc_20260430`

practical best：

- `checkpoint = epoch_15`
- `tau_intervene = 0.4`
- `tau_critical = 0.3`
- `full217 shield_rate = 0.7886`

结果比 A2 原始 all-raw 更差。

### 6.3 实验三：完全关掉 `raw critical override`

对应目录：

- `train_log/v10b_risk_calib_a2_epoch15_no_raw_critical_20260430`

practical best：

- `checkpoint = epoch_15`
- `tau_intervene = 0.3`
- `tau_critical = 0.3`
- `full217 shield_rate = 0.8985`

结果进一步变差。

### 6.4 结论

这三组实验把问题钉得很清楚：

- `raw_v10b_critical_rule` 作为 outer override 的确偏保守
- 但它**不是当前 gate 过大的主因**
- 主因已经转移到：
  - classifier 自己的 `p_intervene / p_critical` 分布太高
  - 再加上 calibration 对低漏检的偏好


## 7. practical calibration：不再只盯住 zero-FN

为了让 calibration 更贴近部署场景，我们随后把 `run_v10b_classifier_calibration.py` 扩展为支持两种选优口径：

- `strict_target`
- `practical_gate`

其中 `practical_gate` 的思路是：

1. 先守住核心协议
   - `collision = 0`
   - `merge / endpoint / safety` 过线
2. 再优先压 `full217_shield_rate`
3. 然后再比较 `false_negative`

即使采用这套更务实的规则，A2 的最优 practical gate 仍然只能到：

- `full217 shield_rate = 0.7089`

与 baseline 的 `0.3332` 差距仍然很大。


## 8. A3 conservative gate：继续做最小重训

### 8.1 设计目的

既然 A2 的输出分布仍然过于激进，我们又做了一轮更保守的 A3：

- 模式：`a3_conservative_gate`
- 核心思路：
  - 降低 `warning-only` 对 `intervene` 的推动
  - 提高 `normal-safe` 的保守约束
  - 适当减轻辅助头影响

### 8.2 A3 的关键改动

`intervene` 标签更保守：

- `intervened -> 1.0`
- `warning-only -> 0.2`
- `normal-safe -> 0.0`

损失权重改为：

- `2.5 * loss_intervene`
- `4.0 * loss_critical`
- `0.05 * loss_side`
- `0.05 * loss_reason`

训练目录：

- `train_log/v10b_risk_classifier_a3_20260430_154200`

### 8.3 A3 训练观察

A3 训练后，classifier-only 的 `predicted_intervention_rate` 确实下降了，  
说明模型本身比 A2 更保守了。

但这还不等于最终部署 gate 更优，所以我们继续做了 focused calibration。


## 9. A3 focused calibration 结果

对应目录：

- `train_log/v10b_risk_calib_a3_epoch10_practical_allraw_20260430`

practical best：

- `checkpoint = epoch_10`
- `tau_intervene = 0.4`
- `tau_critical = 0.3`
- `full217 merge = 0.8618`
- `full217 endpoint = 0.9124`
- `full217 safety = 0.9124`
- `full217 collision = 0`
- `full217 shield_rate = 0.7084`
- `full217 false_negative_rate = 0.0749`
- `hard15 shield_rate = 0.9457`

### 9.1 与 A2 的对比

与 A2 practical best 相比：

- A2: `shield_rate = 0.7089`
- A3: `shield_rate = 0.7084`

也就是说：

- A3 只比 A2 好了 **0.0005 左右**
- 这个改进量级非常小
- 完全不足以支持“用 A3 替代 baseline 部署”


## 10. 从 A-Train-2 到 A3，这条线最终得到的结论

### 10.1 我们做了什么

从 `A-Train-2` 开始，后续实际完成了以下工作：

1. **重训 A2 binary/critical classifier**
2. **完整做 A2 calibration**
3. **分析 teacher v2 数据，检查 critical / intervened / warning 分布**
4. **验证 `tau_critical` 与 `raw critical override` 的影响**
5. **分别测试**
   - `all raw critical`
   - `overlap_or_ttc`
   - `none`
6. **新增 practical calibration 口径**
7. **在 A2 基础上继续做 A3 conservative retrain**
8. **对 A3 再做 focused calibration**

### 10.2 这条线的核心收获

这条线并不是没有产出，反而把问题定位得很明确：

- `A-gate` 可以学成一个高召回风险提示器
- 但很难在不破坏安全协议的前提下，把 `shield_rate` 压到 baseline 附近
- `raw critical override` 不是主矛盾
- 主矛盾是：
  - classifier 输出本身偏高召回
  - 一旦要求 `collision=0 + merge/endpoint/safety` 全守住
  - calibration 最终还是会偏向高开门率配置


## 11. 目前最优方案

截至当前，**最优正式方案仍然是：**

`U220_D230_epoch290 + v10b_leadgap_policy_veto_recovery`

对应结果：

### `full217`

- `merge_success_rate = 0.8618`
- `endpoint_success_rate = 0.9124`
- `safety_success_rate = 0.9124`
- `collision_rate = 0.0000`
- `shield_intervention_rate = 0.3332`

### `hard15`

- `merge_success_rate = 0.8000`
- `endpoint_success_rate = 1.0000`
- `safety_success_rate = 1.0000`
- `collision_rate = 0.0000`
- `shield_intervention_rate = 0.6023`

### 为什么它仍然最优

因为到目前为止：

- A2 practical best：`full217 shield_rate = 0.7089`
- A3 practical best：`full217 shield_rate = 0.7084`

虽然 A-gate 线保持了 `collision=0`，  
但它们的 shield 依赖仍然远高于 baseline 的 `0.3332`，因此**没有一个版本值得替换当前主线部署**。


## 12. 当前建议

从工程收益角度看，当前建议是：

1. **保留现有最优部署方案**
   - `U220_D230_epoch290 + v10b`
2. **把 A2/A3 线视为一次完整的诊断与验证**
   - 它已经回答了“为什么 gate 很难压下来”
3. **短期内不建议继续在同一 A-gate 方向做小修小补**
   - 再微调阈值、再小改 raw critical，收益大概率仍然很小

如果未来还要继续这条线，更值得的新方向不是“再调一点阈值”，而是：

- 改 A-gate 的使用方式
  - 例如让它做风险分层/诊断，而不是直接当开关
- 或重新设计 teacher / label 定义
  - 让 classifier 学到“什么时候真的不需要开门”

