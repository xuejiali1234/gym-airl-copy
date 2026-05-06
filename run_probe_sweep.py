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

CONTROL_ENV = {
    "PROBE_SEED": "44",
    "PROBE_PPO_EPOCHS": "6",
    "PROBE_PPO_MINI_BATCH_SIZE": "256",
    "PROBE_GENERATOR_LR": "8e-5",
    "PROBE_DISCRIMINATOR_LR": "5e-5",
    "PROBE_N_DISC_UPDATES": "5",
    "PROBE_ENT_COEF": "0.005",
    "PROBE_REWARD_NORM": "0",
    "PROBE_SAFETY_UNFREEZE_TIMESTEPS": str(U220_TIMESTEPS),
    "PROBE_SAFETY_LIGHT_UNFREEZE_LR": "5e-6",
    "PROBE_LATE_N_DISC_EPOCH": "230",
    "PROBE_LATE_N_DISC_UPDATES": "4",
    "PROBE_SAFETY_DECAY_EPOCH": "250",
    "PROBE_SAFETY_DECAY_LR": "2.5e-6",
    "PROBE_SAFETY_RAMP_UNFREEZE_EPOCHS": "0",
    "PROBE_SAFETY_DECAY_RAMP_EPOCHS": "0",
    "PROBE_SAFETY_FUSE_FEATURE": "0",
    "PROBE_SAFETY_EMBED_DIM": "1",
    "PROBE_ENABLE_PREDICTIVE_SAFETY_CRITIC": "0",
    "PROBE_PREDICTIVE_SAFETY_HORIZON_STEPS": "10",
    "PROBE_PREDICTIVE_SAFETY_DT": "0.1",
    "PROBE_PREDICTIVE_SAFETY_USE_CANDIDATES": "1",
    "PROBE_PREDICTIVE_SAFETY_GEN_PENALTY": "0.0",
    "PROBE_ENABLE_PREDICTIVE_SAFETY_RESIDUAL": "1",
    "PROBE_PREDICTIVE_SAFETY_RESIDUAL_SCALE": "0.5",
    "PROBE_SAFETY_REG_COEFF": "0.0",
    "PROBE_PREDICTIVE_SAFETY_BASE_REG_COEFF": "0.0",
    "PROBE_PREDICTIVE_SAFETY_LATE_REG_EPOCH": "250",
    "PROBE_PREDICTIVE_SAFETY_LATE_REG_COEFF": "0.0",
    "PROBE_PREDICTIVE_SAFETY_REG_MODE": "legacy_aux",
    "PROBE_PREDICTIVE_SAFETY_ENABLE_CPAIR_ADDITIVE": "1",
    "PROBE_PREDICTIVE_SAFETY_CPAIR_ADDITIVE_START_EPOCH": "1",
    "PROBE_PREDICTIVE_SAFETY_CPAIR_ADDITIVE_COEFF": "0.03",
    "PROBE_PREDICTIVE_SAFETY_CANDIDATE_SET": "current",
    "PROBE_PREDICTIVE_SAFETY_SAFE_SELECTION": "min_risk",
    "PROBE_PREDICTIVE_SAFETY_RANK_METRIC": "clipped",
}

EXPERIMENTS = [
    {
        "name": "P45_P30_SmoothTransitions",
        "description": "P30 base with only schedule smoothing: safety unfreeze ramps over epochs 220-240, D5/D4 transition is spread over 210-230, and Decay250 ramps over 250-270; no reward/objective changes.",
        "env": {
            "PROBE_SAFETY_RAMP_UNFREEZE_EPOCHS": "20",
            "PROBE_LATE_N_DISC_RAMP_START_EPOCH": "210",
            "PROBE_LATE_N_DISC_RAMP_END_EPOCH": "230",
            "PROBE_SAFETY_DECAY_RAMP_EPOCHS": "20",
            "PROBE_BEST_SELECT_START_EPOCH": "270",
            "PROBE_SAVE_FREQ_EPOCHS": "1",
        },
    },
    {
        "name": "P30_CPairD250_NoLateLR_Save1_Rerun",
        "description": "Original P30 rerun for reproducibility: residual0.5, no legacy aux, CPair additive 0.03 from epoch 1, D230 late n_disc=4, Decay250, no generator penalty.",
        "env": {
            "PROBE_SAFETY_RAMP_UNFREEZE_EPOCHS": "0",
            "PROBE_SAFETY_DECAY_RAMP_EPOCHS": "0",
            "PROBE_BEST_SELECT_START_EPOCH": "270",
            "PROBE_SAVE_FREQ_EPOCHS": "1",
        },
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run late-stage AIRL schedule probes sequentially.")
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
    experiments = EXPERIMENTS
    if not args.only:
        return experiments

    requested = [item.strip() for item in args.only.split(",") if item.strip()]
    selected = []
    for exp in experiments:
        if any(exp["name"] == req or exp["name"].startswith(f"{req}_") or exp["name"].startswith(req) for req in requested):
            selected.append(exp)
    if not selected:
        raise ValueError(f"No experiments matched --only={args.only!r}")
    return selected


def build_env(exp):
    env = os.environ.copy()
    overrides = dict(BASE_ENV)
    overrides.update(CONTROL_ENV)
    overrides.update(exp["env"])
    overrides["PROBE_TAG"] = exp["name"]
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

    print("[*] Probe sweep plan:")
    for idx, exp in enumerate(experiments, start=1):
        _, overrides = build_env(exp)
        manifest["experiments"].append(
            {
                "name": exp["name"],
                "description": exp["description"],
                "overrides": overrides,
            }
        )
        print(f"  {idx}. {exp['name']}: {exp['description']}")
        print(f"     overrides={overrides}")

    manifest_path = REPO_ROOT / "train_log" / f"probe_sweep_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
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
        print(f"\n[*] Running probe {idx}/{len(experiments)}: {exp['name']}")
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
            raise SystemExit(f"Probe failed: {exp['name']} (exit={result.returncode})")

    print("\n[*] Probe sweep finished.")


if __name__ == "__main__":
    main()
