# configs/config.py


class Config:
    """
    P30 reference configuration.

    直接运行 train_airl_baseline.py 时，默认训练当前保留主线：
    P30_CPairD250_NoLateLR_Save1

    核心语义：
    - old safety + predictive residual
    - residual_scale = 0.5
    - legacy aux = 0
    - CPair additive = 0.03 from epoch 1
    - U220 light unfreeze
    - D230 discriminator updates 5 -> 4
    - Decay250 safety LR 5e-6 -> 2.5e-6
    - no ramp / no generator penalty / no CandidateV2 / no focused CPair
    """

    # --- Environment ---
    STATE_DIM = 16
    ACTION_DIM = 2
    DT = 0.1

    # Physical limits in ft/s^2.
    PHYS_ACC_MAX = 15.0
    PHYS_STEER_MAX = 8.5
    POLICY_MAX_ACTION = 1.0

    VEHICLE_WIDTH = 6.0
    VEHICLE_LENGTH = 15.0
    LANE_WIDTH = 12.0

    X_MIN = 48
    X_MAX = 72
    GOAL_TOLERANCE = 10.0
    SPEED_LIMIT = 80.0
    PPO_RL_GOAL_PROGRESS_SCALE = 20.0
    PPO_RL_THW_SAFE_SECONDS = 2.0

    # --- Reproducibility ---
    DEVICE = "cuda"
    SEED = 44
    DETERMINISTIC_TRAINING = True

    # --- P30 training base ---
    EPOCHS = 300
    STEPS_PER_EPOCH = 2048
    OUTER_UPDATE_FREQ = 5

    PPO_EPOCHS = 6
    PPO_MINI_BATCH_SIZE = 256
    PPO_ENT_COEF = 0.005

    GENERATOR_LEARNING_RATE = 8e-5
    DISCRIMINATOR_LEARNING_RATE = 5e-5

    USE_TRAINING_EXTENSIONS = True
    CONFIDENCE_WARMUP_EPOCHS = 10
    ENABLE_GOAL_CONDITION = True
    GOAL_ABLATION_MODE = "normal"  # normal | zero | drop
    ENABLE_CONFIDENCE_WEIGHTING = False
    RANKING_MARGIN = 0.2

    ENABLE_BC_PRETRAIN = False
    BC_PRETRAIN_EPOCHS = 70
    BC_LEARNING_RATE = 1e-3

    ENABLE_REWARD_CLIPPING = True
    REWARD_CLIP_MIN = -2.0
    REWARD_CLIP_MAX = 2.0
    ENABLE_REWARD_NORMALIZATION = False
    DEBUG_USE_GROUND_TRUTH_REWARD = False

    ENABLE_ATTENTION = True
    ATTENTION_ABLATION_MODE = "normal"  # normal | zero

    # --- Evaluation / checkpoint cadence ---
    # These names are kept for compatibility with train_airl_baseline.py.
    PROBE_SAVE_FREQ_EPOCHS = 1
    PROBE_QUICK_EVAL_EPISODES = 8
    PROBE_FULL_EVAL_EPISODES = 100
    PROBE_FULL_EVAL_FREQ_EPOCHS = 1
    PROBE_FULL_EVAL_PRE_END_EPOCH = 0
    PROBE_FULL_EVAL_PRE_FREQ_EPOCHS = 1
    PROBE_EPOCH0_EVAL_EPISODES = 100
    PROBE_BEST_SELECT_START_EPOCH = 270

    # --- Discriminator late schedule: D230 ---
    PROBE_N_DISC_UPDATES = 5
    PROBE_LATE_N_DISC_EPOCH = 230
    PROBE_LATE_N_DISC_UPDATES = 4

    # --- Safety module ---
    ENABLE_SAFETY_PRETRAIN = True
    ENABLE_SAFETY_MODULE = True
    ENABLE_SAFETY_BRANCH = True
    ENABLE_SAFETY_AUX_LOSS = True

    SAFETY_LEARNING_RATE = 1e-4
    SAFETY_COEFF = 0.0
    SAFETY_REGULATOR_COEFF = 0.0
    SAFETY_USE_ACTION = True
    SAFETY_FUSE_FEATURE = False
    SAFETY_EMBED_DIM = 1
    SAFETY_BATCH_SIZE = 256

    # U220: keep safety frozen until epoch 220, then light-finetune it.
    SAFETY_UNFREEZE_TIMESTEPS = 220 * STEPS_PER_EPOCH
    SAFETY_LIGHT_UNFREEZE_LR = 5e-6

    # Decay250: after epoch 250, halve the safety light-finetune LR.
    PROBE_SAFETY_DECAY_EPOCH = 250
    PROBE_SAFETY_DECAY_LR = 2.5e-6

    # --- Predictive residual + CPair additive ---
    ENABLE_PREDICTIVE_SAFETY_CRITIC = False
    ENABLE_PREDICTIVE_SAFETY_RESIDUAL = True
    PREDICTIVE_SAFETY_HORIZON_STEPS = 10
    PREDICTIVE_SAFETY_DT = 0.1
    PREDICTIVE_SAFETY_USE_CANDIDATES = True
    PREDICTIVE_SAFETY_RESIDUAL_SCALE = 0.5

    # Legacy aux is intentionally off in P30.
    PREDICTIVE_SAFETY_REG_MODE = "legacy_aux"
    PREDICTIVE_SAFETY_REG_MARGIN = 0.2
    PREDICTIVE_SAFETY_BASE_REG_COEFF = 0.0

    # CPair additive is the active discriminator-side safety shaper.
    PREDICTIVE_SAFETY_ENABLE_CPAIR_ADDITIVE = True
    PREDICTIVE_SAFETY_CPAIR_ADDITIVE_START_EPOCH = 1
    PREDICTIVE_SAFETY_CPAIR_ADDITIVE_COEFF = 0.03
    PREDICTIVE_SAFETY_CANDIDATE_SET = "current"
    PREDICTIVE_SAFETY_SAFE_SELECTION = "min_risk"
    PREDICTIVE_SAFETY_RANK_METRIC = "clipped"

    # Keep the calibrated speed-aware gap formula used by the current safety residual.
    PREDICTIVE_GAP_LEAD_TAU = 0.5
    PREDICTIVE_GAP_FOLLOW_TAU = 0.4

    # --- Normalization placeholders ---
    # Actual data-driven normalization is handled by data_loader.py / main training flow.
    STATE_SCALE = None
    STATE_BIAS = None
