import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "RL基线对比" / "train_ppo_rl_baseline.py"

runpy.run_path(str(TARGET), run_name="__main__")
