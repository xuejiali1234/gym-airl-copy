import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
STEPS_PER_EPOCH = 2048
U220_TIMESTEPS = 220 * STEPS_PER_EPOCH

BASE_ENV = {
    "PROBE_EPOCHS": "300",
    "PROBE_SAVE_FREQ_EPOCHS": "1",
    "PROBE_QUICK_EVAL_EPISODES": "8",
    "PROBE_FULL_EVAL_EPISODES": "100",
    "PROBE_FULL_EVAL_PRE_END_EPOCH": "0",
    "PROBE_FULL_EVAL_PRE_FREQ_EPOCHS": "1",
    "PROBE_FULL_EVAL_FREQ_EPOCHS": "1",
    "PROBE_EPOCH0_EVAL_EPISODES": "100",
    "PROBE_REWARD_NORM": "0",
    "PROBE_BEST_SELECT_START_EPOCH": "270",
    "PROBE_SEED": "44",
    "PROBE_PPO_EPOCHS": "6",
    "PROBE_PPO_MINI_BATCH_SIZE": "256",
    "PROBE_GENERATOR_LR": "8e-5",
    "PROBE_DISCRIMINATOR_LR": "5e-5",
    "PROBE_N_DISC_UPDATES": "5",
    "PROBE_ENT_COEF": "0.005",
    "PROBE_LATE_N_DISC_EPOCH": "230",
    "PROBE_LATE_N_DISC_UPDATES": "4",
    "PROBE_SAFETY_RAMP_UNFREEZE_EPOCHS": "0",
    "PROBE_SAFETY_DECAY_RAMP_EPOCHS": "0",
    "PROBE_ENABLE_PREDICTIVE_SAFETY_CRITIC": "0",
    "PROBE_PREDICTIVE_SAFETY_HORIZON_STEPS": "10",
    "PROBE_PREDICTIVE_SAFETY_DT": "0.1",
    "PROBE_PREDICTIVE_SAFETY_USE_CANDIDATES": "1",
    "PROBE_PREDICTIVE_SAFETY_GEN_PENALTY": "0.0",
    "PROBE_SAFETY_REG_COEFF": "0.0",
    "PROBE_PREDICTIVE_SAFETY_BASE_REG_COEFF": "0.0",
    "PROBE_PREDICTIVE_SAFETY_LATE_REG_EPOCH": "250",
    "PROBE_PREDICTIVE_SAFETY_LATE_REG_COEFF": "0.0",
    "PROBE_PREDICTIVE_SAFETY_REG_MODE": "legacy_aux",
    "PROBE_PREDICTIVE_SAFETY_CANDIDATE_SET": "current",
    "PROBE_PREDICTIVE_SAFETY_SAFE_SELECTION": "min_risk",
    "PROBE_PREDICTIVE_SAFETY_RANK_METRIC": "clipped",
}

SAFETY_ON_ENV = {
    "PROBE_ENABLE_SAFETY_MODULE": "1",
    "PROBE_ENABLE_SAFETY_BRANCH": "1",
    "PROBE_ENABLE_SAFETY_AUX_LOSS": "1",
    "PROBE_SAFETY_FUSE_FEATURE": "0",
    "PROBE_SAFETY_EMBED_DIM": "1",
    "PROBE_SAFETY_UNFREEZE_TIMESTEPS": str(U220_TIMESTEPS),
    "PROBE_SAFETY_LIGHT_UNFREEZE_LR": "5e-6",
    "PROBE_SAFETY_DECAY_EPOCH": "250",
    "PROBE_SAFETY_DECAY_LR": "2.5e-6",
    "PROBE_ENABLE_PREDICTIVE_SAFETY_RESIDUAL": "1",
    "PROBE_PREDICTIVE_SAFETY_RESIDUAL_SCALE": "0.5",
    "PROBE_PREDICTIVE_SAFETY_ENABLE_CPAIR_ADDITIVE": "1",
    "PROBE_PREDICTIVE_SAFETY_CPAIR_ADDITIVE_START_EPOCH": "1",
    "PROBE_PREDICTIVE_SAFETY_CPAIR_ADDITIVE_COEFF": "0.03",
}

SAFETY_OFF_ENV = {
    "PROBE_ENABLE_SAFETY_MODULE": "0",
    "PROBE_ENABLE_SAFETY_BRANCH": "0",
    "PROBE_ENABLE_SAFETY_AUX_LOSS": "0",
    "PROBE_SAFETY_FUSE_FEATURE": "0",
    "PROBE_SAFETY_EMBED_DIM": "1",
    "PROBE_SAFETY_UNFREEZE_TIMESTEPS": str(U220_TIMESTEPS),
    "PROBE_SAFETY_LIGHT_UNFREEZE_LR": "5e-6",
    "PROBE_SAFETY_DECAY_EPOCH": "250",
    "PROBE_SAFETY_DECAY_LR": "2.5e-6",
    "PROBE_ENABLE_PREDICTIVE_SAFETY_RESIDUAL": "0",
    "PROBE_PREDICTIVE_SAFETY_RESIDUAL_SCALE": "0.5",
    "PROBE_PREDICTIVE_SAFETY_ENABLE_CPAIR_ADDITIVE": "0",
    "PROBE_PREDICTIVE_SAFETY_CPAIR_ADDITIVE_START_EPOCH": "1",
    "PROBE_PREDICTIVE_SAFETY_CPAIR_ADDITIVE_COEFF": "0.0",
}

COMBINATIONS = [
    ("G0A0S0_AIRL_Base", False, False, False, "Vanilla AIRL baseline: no explicit goal, no attention, no safety module."),
    ("G1A0S0_Goal", True, False, False, "AIRL + explicit goal only."),
    ("G0A1S0_Attn", False, True, False, "AIRL + attention only."),
    ("G0A0S1_Safety", False, False, True, "AIRL + P30 safety module only."),
    ("G1A1S0_Goal_Attn", True, True, False, "AIRL + explicit goal + attention."),
    ("G1A0S1_Goal_Safety", True, False, True, "AIRL + explicit goal + P30 safety module."),
    ("G0A1S1_Attn_Safety", False, True, True, "AIRL + attention + P30 safety module."),
    ("G1A1S1_P30_Full", True, True, True, "Full P30 model: explicit goal + attention + P30 safety module."),
]

EXPERIMENTS = [
    {
        "name": name,
        "goal": goal_enabled,
        "attention": attention_enabled,
        "safety": safety_enabled,
        "description": description,
    }
    for name, goal_enabled, attention_enabled, safety_enabled, description in COMBINATIONS
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run coarse AIRL module combinations: goal, attention, safety.")
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated experiment prefixes/names to run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned experiments without launching training.",
    )
    return parser.parse_args()


def selected_experiments(args):
    if not args.only:
        return EXPERIMENTS

    requested = [item.strip() for item in args.only.split(",") if item.strip()]
    selected = []
    for exp in EXPERIMENTS:
        if any(exp["name"] == req or exp["name"].startswith(f"{req}_") or exp["name"].startswith(req) for req in requested):
            selected.append(exp)
    if not selected:
        raise ValueError(f"No experiments matched --only={args.only!r}")
    return selected


def build_env(exp):
    overrides = dict(BASE_ENV)
    overrides.update(SAFETY_ON_ENV if exp["safety"] else SAFETY_OFF_ENV)
    overrides.update(
        {
            "PROBE_ENABLE_GOAL_CONDITION": "1" if exp["goal"] else "0",
            "PROBE_GOAL_ABLATION_MODE": "normal" if exp["goal"] else "drop",
            "PROBE_ENABLE_ATTENTION": "1" if exp["attention"] else "0",
            "PROBE_ATTENTION_ABLATION_MODE": "normal",
            "PROBE_TAG": exp["name"],
        }
    )

    env = os.environ.copy()
    env.update(overrides)
    env["PYTHONUNBUFFERED"] = "1"
    return env, overrides


def main():
    args = parse_args()
    experiments = selected_experiments(args)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python": sys.executable,
        "experiments": [],
    }

    print("[*] AIRL module combination sweep plan:")
    for idx, exp in enumerate(experiments, start=1):
        _, overrides = build_env(exp)
        manifest["experiments"].append(
            {
                "name": exp["name"],
                "goal": exp["goal"],
                "attention": exp["attention"],
                "safety": exp["safety"],
                "description": exp["description"],
                "overrides": overrides,
            }
        )
        print(
            f"  {idx}. {exp['name']}: "
            f"G={int(exp['goal'])}, A={int(exp['attention'])}, S={int(exp['safety'])} | "
            f"{exp['description']}"
        )
        print(f"     overrides={overrides}")

    manifest_path = REPO_ROOT / "train_log" / f"airl_module_combo_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    child_log_dir = manifest_path.parent / f"{manifest_path.stem}_logs"
    child_log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Saved sweep manifest: {manifest_path}")

    if args.dry_run:
        print("[*] Dry run only. No training launched.")
        return

    for idx, exp in enumerate(experiments, start=1):
        env, _ = build_env(exp)
        stdout_path = child_log_dir / f"{idx:02d}_{exp['name']}.out.log"
        stderr_path = child_log_dir / f"{idx:02d}_{exp['name']}.err.log"
        print(f"\n[*] Running module combination {idx}/{len(experiments)}: {exp['name']}")
        print(f"    stdout: {stdout_path}")
        print(f"    stderr: {stderr_path}")
        with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout_file, stderr_path.open(
            "w", encoding="utf-8", errors="replace"
        ) as stderr_file:
            result = subprocess.run(
                [sys.executable, "train_airl_baseline.py"],
                cwd=str(REPO_ROOT),
                env=env,
                stdout=stdout_file,
                stderr=stderr_file,
            )
        if result.returncode != 0:
            raise SystemExit(f"Module combination failed: {exp['name']} (exit={result.returncode})")

    print("\n[*] AIRL module combination sweep finished.")


if __name__ == "__main__":
    main()
