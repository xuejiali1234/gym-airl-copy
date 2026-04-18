import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent

BASE_ENV = {
    "PROBE_EPOCHS": "100",
    "PROBE_SAVE_FREQ_EPOCHS": "10",
    "PROBE_QUICK_EVAL_EPISODES": "4",
    "PROBE_FULL_EVAL_EPISODES": "10",
    "PROBE_REWARD_NORM": "0",
}

EXPERIMENTS = [
    {
        "name": "A_base_ppo8_glr8e5",
        "description": "Current short baseline: PPO_EPOCHS=8, generator_lr=8e-5.",
        "env": {
            "PROBE_PPO_EPOCHS": "8",
            "PROBE_GENERATOR_LR": "8e-5",
        },
    },
    {
        "name": "B_ppo6_glr8e5",
        "description": "Lower PPO epochs to test more stable generator updates.",
        "env": {
            "PROBE_PPO_EPOCHS": "6",
            "PROBE_GENERATOR_LR": "8e-5",
        },
    },
    {
        "name": "C_ppo6_glr7e5",
        "description": "Lower PPO epochs and generator learning rate.",
        "env": {
            "PROBE_PPO_EPOCHS": "6",
            "PROBE_GENERATOR_LR": "7e-5",
        },
    },
    {
        "name": "D_ppo8_glr7e5",
        "description": "Keep PPO epochs, lower generator learning rate.",
        "env": {
            "PROBE_PPO_EPOCHS": "8",
            "PROBE_GENERATOR_LR": "7e-5",
        },
    },
    {
        "name": "E_ppo6_glr7e5_rnorm",
        "description": "Probe C plus reward normalization.",
        "env": {
            "PROBE_PPO_EPOCHS": "6",
            "PROBE_GENERATOR_LR": "7e-5",
            "PROBE_REWARD_NORM": "1",
        },
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run short AIRL probe sweeps sequentially.")
    parser.add_argument(
        "--include-rnorm",
        action="store_true",
        help="Also run experiment E with reward normalization.",
    )
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated experiment prefixes/names to run, e.g. A,B or C_ppo6_glr7e5.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned experiments without launching training.",
    )
    return parser.parse_args()


def selected_experiments(args):
    experiments = EXPERIMENTS if args.include_rnorm else [exp for exp in EXPERIMENTS if not exp["name"].startswith("E_")]
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
