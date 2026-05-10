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
}


P30_STRUCTURE_ENV = {
    "PROBE_SEED": "44",
    "PROBE_PPO_EPOCHS": "6",
    "PROBE_PPO_MINI_BATCH_SIZE": "256",
    "PROBE_GENERATOR_LR": "8e-5",
    "PROBE_DISCRIMINATOR_LR": "5e-5",
    "PROBE_N_DISC_UPDATES": "5",
    "PROBE_ENT_COEF": "0.005",
    "PROBE_REWARD_NORM": "0",
    "PROBE_ENABLE_GOAL_CONDITION": "1",
    "PROBE_ENABLE_ATTENTION": "1",
    "PROBE_ENABLE_SAFETY_MODULE": "1",
    "PROBE_ENABLE_SAFETY_AUX_LOSS": "1",
    "PROBE_SAFETY_FUSE_FEATURE": "0",
    "PROBE_SAFETY_EMBED_DIM": "1",
    "PROBE_SAFETY_UNFREEZE_TIMESTEPS": str(U220_TIMESTEPS),
    "PROBE_SAFETY_LIGHT_UNFREEZE_LR": "5e-6",
    "PROBE_LATE_N_DISC_EPOCH": "230",
    "PROBE_LATE_N_DISC_UPDATES": "4",
    "PROBE_SAFETY_DECAY_EPOCH": "250",
    "PROBE_SAFETY_DECAY_LR": "2.5e-6",
    "PROBE_SAFETY_RAMP_UNFREEZE_EPOCHS": "0",
    "PROBE_SAFETY_DECAY_RAMP_EPOCHS": "0",
    "PROBE_ENABLE_PREDICTIVE_SAFETY_CRITIC": "0",
    "PROBE_PREDICTIVE_SAFETY_HORIZON_STEPS": "10",
    "PROBE_PREDICTIVE_SAFETY_DT": "0.1",
    "PROBE_PREDICTIVE_SAFETY_USE_CANDIDATES": "1",
    "PROBE_PREDICTIVE_SAFETY_GEN_PENALTY": "0.0",
    "PROBE_ENABLE_PREDICTIVE_SAFETY_RESIDUAL": "1",
    "PROBE_SAFETY_REG_COEFF": "0.0",
    "PROBE_PREDICTIVE_SAFETY_BASE_REG_COEFF": "0.0",
    "PROBE_PREDICTIVE_SAFETY_LATE_REG_EPOCH": "250",
    "PROBE_PREDICTIVE_SAFETY_LATE_REG_COEFF": "0.0",
    "PROBE_PREDICTIVE_SAFETY_REG_MODE": "legacy_aux",
    "PROBE_PREDICTIVE_SAFETY_CPAIR_ADDITIVE_START_EPOCH": "1",
    "PROBE_PREDICTIVE_SAFETY_CANDIDATE_SET": "current",
    "PROBE_PREDICTIVE_SAFETY_SAFE_SELECTION": "min_risk",
    "PROBE_PREDICTIVE_SAFETY_RANK_METRIC": "clipped",
}


SIGNAL_ON_SAFETY_ENV = {
    "PROBE_ENABLE_SAFETY_BRANCH": "1",
    "PROBE_PREDICTIVE_SAFETY_RESIDUAL_SCALE": "0.5",
    "PROBE_PREDICTIVE_SAFETY_ENABLE_CPAIR_ADDITIVE": "1",
    "PROBE_PREDICTIVE_SAFETY_CPAIR_ADDITIVE_COEFF": "0.03",
}


SIGNAL_ZERO_SAFETY_ENV = {
    "PROBE_ENABLE_SAFETY_BRANCH": "0",
    "PROBE_PREDICTIVE_SAFETY_RESIDUAL_SCALE": "0.0",
    "PROBE_PREDICTIVE_SAFETY_ENABLE_CPAIR_ADDITIVE": "0",
    "PROBE_PREDICTIVE_SAFETY_CPAIR_ADDITIVE_COEFF": "0.0",
}


COMBINATIONS = [
    (
        "SZ_G0A0S0_AllZero",
        False,
        False,
        False,
        "P30 architecture with goal zeroed, attention output zeroed, and safety signals zeroed.",
    ),
    (
        "SZ_G1A0S0_GoalOnly",
        True,
        False,
        False,
        "P30 architecture with only the goal signal active.",
    ),
    (
        "SZ_G0A1S0_AttnOnly",
        False,
        True,
        False,
        "P30 architecture with only the attention signal active.",
    ),
    (
        "SZ_G0A0S1_SafetyOnly",
        False,
        False,
        True,
        "P30 architecture with only the safety signals active.",
    ),
    (
        "SZ_G1A1S0_Goal_Attn",
        True,
        True,
        False,
        "P30 architecture with goal and attention active, safety signals zeroed.",
    ),
    (
        "SZ_G1A0S1_Goal_Safety",
        True,
        False,
        True,
        "P30 architecture with goal and safety active, attention output zeroed.",
    ),
    (
        "SZ_G0A1S1_Attn_Safety",
        False,
        True,
        True,
        "P30 architecture with attention and safety active, goal zeroed.",
    ),
    (
        "SZ_G1A1S1_P30_Full",
        True,
        True,
        True,
        "Sanity control: full P30 architecture and all signals active.",
    ),
]


EXPERIMENTS = [
    {
        "name": name,
        "goal_signal": goal_signal,
        "attention_signal": attention_signal,
        "safety_signal": safety_signal,
        "description": description,
    }
    for name, goal_signal, attention_signal, safety_signal, description in COMBINATIONS
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run strict signal-zero module ablations on the fixed P30 architecture."
    )
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
        if any(
            exp["name"] == req or exp["name"].startswith(f"{req}_") or exp["name"].startswith(req)
            for req in requested
        ):
            selected.append(exp)
    if not selected:
        raise ValueError(f"No experiments matched --only={args.only!r}")
    return selected


def build_overrides(exp):
    overrides = dict(BASE_ENV)
    overrides.update(P30_STRUCTURE_ENV)
    overrides.update(
        {
            "PROBE_GOAL_ABLATION_MODE": "normal" if exp["goal_signal"] else "zero",
            "PROBE_ATTENTION_ABLATION_MODE": "normal" if exp["attention_signal"] else "zero",
        }
    )
    overrides.update(SIGNAL_ON_SAFETY_ENV if exp["safety_signal"] else SIGNAL_ZERO_SAFETY_ENV)
    overrides["PROBE_TAG"] = exp["name"]
    return overrides


def build_env(exp):
    overrides = build_overrides(exp)
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
        "note": "Strict signal-zero ablation: P30 architecture is preserved in all runs.",
        "experiments": [],
    }

    print("[*] Signal-zero module ablation sweep plan:")
    for idx, exp in enumerate(experiments, start=1):
        _, overrides = build_env(exp)
        manifest["experiments"].append(
            {
                "name": exp["name"],
                "goal_signal": exp["goal_signal"],
                "attention_signal": exp["attention_signal"],
                "safety_signal": exp["safety_signal"],
                "description": exp["description"],
                "overrides": overrides,
            }
        )
        print(
            f"  {idx}. {exp['name']}: "
            f"G={int(exp['goal_signal'])}, A={int(exp['attention_signal'])}, "
            f"S={int(exp['safety_signal'])} | {exp['description']}"
        )
        print(f"     overrides={overrides}")

    manifest_path = REPO_ROOT / "train_log" / (
        f"signal_zero_module_ablation_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
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
        print(f"\n[*] Running signal-zero ablation {idx}/{len(experiments)}: {exp['name']}")
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
            raise SystemExit(f"Signal-zero ablation failed: {exp['name']} (exit={result.returncode})")

    print("\n[*] Signal-zero module ablation sweep finished.")


if __name__ == "__main__":
    main()
