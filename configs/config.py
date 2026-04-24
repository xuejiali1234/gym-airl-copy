# configs/config.py

class Config:
    """
    全局配置类 (Clean Version)
    只保留核心物理参数和训练参数，移除了所有辅助奖励和限制开关。
    """
    # --- 环境参数 ---
    STATE_DIM = 16
    ACTION_DIM = 2
    DT = 0.1

    # [物理极限]
    # 纵向 (ft/s^2)
    PHYS_ACC_MAX = 15.0
    # 横向 (ft/s^2)
    PHYS_STEER_MAX = 8.5

    # [策略归一化] Policy 输出 [-1, 1]
    POLICY_MAX_ACTION = 1.0

    # 车辆物理尺寸
    VEHICLE_WIDTH = 6.0
    VEHICLE_LENGTH = 15.0
    LANE_WIDTH = 12.0

    # 道路边界
    X_MIN = 48
    X_MAX = 72

    # [新增] PPO 训练参数
    PPO_EPOCHS = 6
    PPO_MINI_BATCH_SIZE = 256

    # [新增] Ranking Loss 的 Margin 建议设为 0.1 到 0.5 之间。
    # 如果 Value Network 的输出本身就在 0.0~1.0 之间，设 1.0 太难了，设 0.2 比较合理。
    RANKING_MARGIN = 0.2

    # 动态终点判定参数
    GOAL_TOLERANCE = 10.0
    
    # [新增] 最大速度限制 (ft/s)
    # NGSIM 场景通常在 40-80 ft/s，100.0 是一个安全的上限
    SPEED_LIMIT = 80.0

    # --- 训练参数 ---
    DEVICE = "cuda"
    SEED = 44
    DETERMINISTIC_TRAINING = True
    EPOCHS = 300
    STEPS_PER_EPOCH = 2048
    OUTER_UPDATE_FREQ = 5

    # --- 训练增强 ---
    USE_TRAINING_EXTENSIONS = True
    CONFIDENCE_WARMUP_EPOCHS = 10

    # BC (虽然设为False，但保留参数防报错)
    ENABLE_BC_PRETRAIN = False
    BC_PRETRAIN_EPOCHS = 70
    BC_LEARNING_RATE = 1e-3

    ENABLE_REWARD_CLIPPING = True
    REWARD_CLIP_MIN = -2.0
    REWARD_CLIP_MAX = 2.0

    # [安全预训练]
    ENABLE_SAFETY_PRETRAIN = True

    # 学习率
    # Reference reproduction target: baseline_attn_20260412_183104
    # Alternative safety-heavier reference: baseline_attn_20260412_192958 -> 7e-5
    DISCRIMINATOR_LEARNING_RATE = 5e-5
    GENERATOR_LEARNING_RATE = 8e-5

    # [新增] 消融实验开关 (Ablation Study)
    # -------------------------------------------------
    # 是否使用目标引导 (G):
    # True = 输入 State + Goal; False = 仅输入 State
    ENABLE_GOAL_CONDITION = True

    # 是否使用置信度加权/Ranking Loss (C):
    # True = 使用 Ranking Loss 和 Beta 加权; False = 标准 AIRL (无加权)
    ENABLE_CONFIDENCE_WEIGHTING = False
    # -------------------------------------------------

    # [新增] 安全机制 (S-AIRL) 消融实验
    # -------------------------------------------------
    # True = 启用安全网络及安全惩罚; False = 关闭 (标准 GC-AIRL)
    ENABLE_SAFETY_MODULE = True
    # True = 使用真实预训练安全先验; False = 使用恒零安全先验以做结构保持消融
    ENABLE_SAFETY_BRANCH = True
    # True = 启用判别器辅助安全损失; False = 仅保留安全先验融合分支
    ENABLE_SAFETY_AUX_LOSS = True

    # [新增] 注意力机制消融实验开关
    # True = 开启多头注意力机制; False = 关闭 (返回全 0 特征)
    ENABLE_ATTENTION = True
    # Stabilization/debug switches. Defaults are off to keep the tuned full model unchanged.
    # 判别器奖励归一化
    ENABLE_REWARD_NORMALIZATION = False
    # 完全使用环境奖励
    DEBUG_USE_GROUND_TRUTH_REWARD = False

    

    # 安全模块超参数
    SAFETY_LEARNING_RATE = 1e-4
    SAFETY_COEFF = 0.00       # Lambda (惩罚权重)
    SAFETY_REGULATOR_COEFF = 0.06  # [新增] 安全调节器权重
    SAFETY_USE_ACTION = True
    SAFETY_FUSE_FEATURE = True
    SAFETY_BATCH_SIZE = 256
    # First stabilize training with a fully frozen safety prior.
    SAFETY_UNFREEZE_TIMESTEPS = EPOCHS * STEPS_PER_EPOCH + 1
    SAFETY_LIGHT_UNFREEZE_LR = 1e-5
    # -------------------------------------------------


    # --- 归一化参数 (DEPRECATED) ---
    # Moved to Data-Driven Normalization in data_loader.py and main.py
    STATE_SCALE = None
    STATE_BIAS = None

    # Old values kept for reference if needed
    # STATE_SCALE = [ ... ]
    # STATE_BIAS = [ ... ]
