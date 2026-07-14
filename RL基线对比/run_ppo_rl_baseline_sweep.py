import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent

BASE_ENV = {
    "PPO_RL_EPOCHS": "300",
    "PPO_RL_SAVE_FREQ_EPOCHS": "1",
    "PPO_RL_QUICK_EVAL_EPISODES": "8",
    "PPO_RL_FULL_EVAL_EPISODES": "100",
    "PPO_RL_FULL_EVAL_FREQ_EPOCHS": "1",
    "PPO_RL_EPOCH0_EVAL_EPISODES": "100",
    "PPO_RL_BEST_SELECT_START_EPOCH": "270",
}

CONTROL_ENV = {
    "PPO_RL_PPO_EPOCHS": "6",
    "PPO_RL_PPO_MINI_BATCH_SIZE": "256",
    "PPO_RL_ENT_COEF": "0.005",
    "PPO_RL_LR": "8e-5",
    "PPO_RL_GAMMA": "0.99",
    "PPO_RL_GAE_LAMBDA": "0.95",
    "PPO_RL_CLIP_RANGE": "0.2",
    "PPO_RL_VF_COEF": "0.5",
    "PPO_RL_MAX_GRAD_NORM": "0.5",
    "PPO_RL_REWARD_CLIP_MIN": "-10.0",
    "PPO_RL_REWARD_CLIP_MAX": "3.0",
    "PPO_RL_USE_GOAL": "1",
    "PPO_RL_USE_ATTENTION": "0",
    "PPO_RL_W_EFF": "0.20",
    "PPO_RL_W_SAFETY": "1.00",
    "PPO_RL_W_THW": "0.0",
    "PPO_RL_W_COMFORT": "0.05",
    "PPO_RL_W_GOAL": "0.80",
    "PPO_RL_COLLISION_PENALTY": "-5.0",
    "PPO_RL_SUCCESS_BONUS": "1.0",
    "PPO_RL_MERGE_BONUS": "0.5",
    "PPO_RL_TIMEOUT_PENALTY": "-1.0",
}

EXPERIMENTS = [
    {
        "name": "PPO_RL_Goal11_Coll45",
        "description": "Goal11 local probe: keep the current best structure and mildly strengthen the one-shot collision penalty from -4.0 to -4.5.",
        "env": {
            "PPO_RL_W_SAFETY": "0.75",
            "PPO_RL_W_THW": "0.0",
            "PPO_RL_W_COMFORT": "0.05",
            "PPO_RL_W_GOAL": "1.10",
            "PPO_RL_COLLISION_PENALTY": "-4.5",
            "PPO_RL_SUCCESS_BONUS": "0.5",
            "PPO_RL_MERGE_BONUS": "0.3",
            "PPO_RL_TIMEOUT_PENALTY": "-2.0",
            "PPO_RL_GOAL_PROGRESS_SCALE": "20.0",
        },
    },
    {
        "name": "PPO_RL_Goal11_Coll5",
        "description": "Goal11 local probe: keep the same structure and strengthen the one-shot collision penalty from -4.0 to -5.0.",
        "env": {
            "PPO_RL_W_SAFETY": "0.75",
            "PPO_RL_W_THW": "0.0",
            "PPO_RL_W_COMFORT": "0.05",
            "PPO_RL_W_GOAL": "1.10",
            "PPO_RL_COLLISION_PENALTY": "-5.0",
            "PPO_RL_SUCCESS_BONUS": "0.5",
            "PPO_RL_MERGE_BONUS": "0.3",
            "PPO_RL_TIMEOUT_PENALTY": "-2.0",
            "PPO_RL_GOAL_PROGRESS_SCALE": "20.0",
        },
    },
    {
        "name": "PPO_RL_Goal11_Safe08",
        "description": "Goal11 local probe: gently raise the continuous TTC safety weight from 0.75 to 0.80 without touching terminal shaping.",
        "env": {
            "PPO_RL_W_SAFETY": "0.80",
            "PPO_RL_W_THW": "0.0",
            "PPO_RL_W_COMFORT": "0.05",
            "PPO_RL_W_GOAL": "1.10",
            "PPO_RL_COLLISION_PENALTY": "-4.0",
            "PPO_RL_SUCCESS_BONUS": "0.5",
            "PPO_RL_MERGE_BONUS": "0.3",
            "PPO_RL_TIMEOUT_PENALTY": "-2.0",
            "PPO_RL_GOAL_PROGRESS_SCALE": "20.0",
        },
    },
    {
        "name": "PPO_RL_Goal11_Safe08_Coll45",
        "description": "Goal11 local probe: combine a mild TTC safety increase with a mild collision-penalty increase, aiming to lower collision without collapsing merge.",
        "env": {
            "PPO_RL_W_SAFETY": "0.80",
            "PPO_RL_W_THW": "0.0",
            "PPO_RL_W_COMFORT": "0.05",
            "PPO_RL_W_GOAL": "1.10",
            "PPO_RL_COLLISION_PENALTY": "-4.5",
            "PPO_RL_SUCCESS_BONUS": "0.5",
            "PPO_RL_MERGE_BONUS": "0.3",
            "PPO_RL_TIMEOUT_PENALTY": "-2.0",
            "PPO_RL_GOAL_PROGRESS_SCALE": "20.0",
        },
    },
    {
        "name": "PPO_RL_Goal105_Safe08_Coll45",
        "description": "Goal11 local probe: lower goal pull one more notch to 1.05 while keeping the mild safety and collision boosts.",
        "env": {
            "PPO_RL_W_SAFETY": "0.80",
            "PPO_RL_W_THW": "0.0",
            "PPO_RL_W_COMFORT": "0.05",
            "PPO_RL_W_GOAL": "1.05",
            "PPO_RL_COLLISION_PENALTY": "-4.5",
            "PPO_RL_SUCCESS_BONUS": "0.5",
            "PPO_RL_MERGE_BONUS": "0.3",
            "PPO_RL_TIMEOUT_PENALTY": "-2.0",
            "PPO_RL_GOAL_PROGRESS_SCALE": "20.0",
        },
    },
    {
        "name": "PPO_RL_Goal105_Safe08_Coll45_Comfort06",
        "description": "Goal11 local probe: lower goal pull, keep the mild safety boost, and slightly raise comfort regularization to smooth merges without using a hard low-push setup.",
        "env": {
            "PPO_RL_W_SAFETY": "0.80",
            "PPO_RL_W_THW": "0.0",
            "PPO_RL_W_COMFORT": "0.06",
            "PPO_RL_COLLISION_PENALTY": "-4.5",
            "PPO_RL_W_GOAL": "1.05",
            "PPO_RL_SUCCESS_BONUS": "0.5",
            "PPO_RL_MERGE_BONUS": "0.25",
            "PPO_RL_TIMEOUT_PENALTY": "-2.0",
            "PPO_RL_GOAL_PROGRESS_SCALE": "22.0",
        },
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run PPO-RL baseline sweeps sequentially.")
    parser.add_argument("--only", default="", help="Comma-separated experiment prefixes/names to run.")
    parser.add_argument("--seeds", default="44", help="Comma-separated seeds. Default: 44")
    parser.add_argument("--epochs", type=int, default=None, help="Optional epoch override for smoke runs.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs without launching training.")
    return parser.parse_args()


def selected_experiments(args):
    if not args.only:
        return EXPERIMENTS
    requested = [item.strip() for item in args.only.split(",") if item.strip()]
    selected = []
    for exp in EXPERIMENTS:
        if any(exp["name"] == req or exp["name"].startswith(req) for req in requested):
            selected.append(exp)
    if not selected:
        raise ValueError(f"No experiments matched --only={args.only!r}")
    return selected


def parse_seeds(raw):
    seeds = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def build_tag(exp_name, seed, multi_seed):
    if multi_seed or seed != 44:
        return f"{exp_name}_S{seed}"
    return exp_name


def build_job_env(exp, seed, tag, epochs_override):
    env = os.environ.copy()
    overrides = dict(BASE_ENV)
    overrides.update(CONTROL_ENV)
    overrides.update(exp["env"])
    overrides["PPO_RL_SEED"] = str(seed)
    overrides["PPO_RL_TAG"] = tag
    if epochs_override is not None:
        overrides["PPO_RL_EPOCHS"] = str(epochs_override)
    env.update(overrides)
    env["PYTHONUNBUFFERED"] = "1"
    return env, overrides


def main():
    args = parse_args()
    experiments = selected_experiments(args)
    seeds = parse_seeds(args.seeds)
    multi_seed = len(seeds) > 1
    jobs = []

    for exp in experiments:
        for seed in seeds:
            tag = build_tag(exp["name"], seed, multi_seed)
            env, overrides = build_job_env(exp, seed, tag, args.epochs)
            jobs.append(
                {
                    "name": exp["name"],
                    "tag": tag,
                    "seed": seed,
                    "description": exp["description"],
                    "env": env,
                    "overrides": overrides,
                }
            )

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python": sys.executable,
        "jobs": [
            {
                "name": job["name"],
                "tag": job["tag"],
                "seed": job["seed"],
                "description": job["description"],
                "overrides": job["overrides"],
            }
            for job in jobs
        ],
    }

    print("[*] PPO-RL baseline sweep plan:")
    for idx, job in enumerate(jobs, start=1):
        print(f"  {idx}. {job['tag']}: {job['description']}")
        print(f"     overrides={job['overrides']}")

    manifest_path = ROOT_DIR / "train_log" / f"ppo_rl_baseline_sweep_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    child_log_dir = manifest_path.parent / f"{manifest_path.stem}_logs"
    child_log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Saved sweep manifest: {manifest_path}")

    if args.dry_run:
        print("[*] Dry run only. No training launched.")
        return

    train_script = SCRIPT_DIR / "train_ppo_rl_baseline.py"
    for idx, job in enumerate(jobs, start=1):
        stdout_path = child_log_dir / f"{idx:02d}_{job['tag']}.out.log"
        stderr_path = child_log_dir / f"{idx:02d}_{job['tag']}.err.log"
        print(f"\n[*] Running PPO-RL job {idx}/{len(jobs)}: {job['tag']}")
        print(f"    stdout: {stdout_path}")
        print(f"    stderr: {stderr_path}")
        with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout_file, stderr_path.open(
            "w", encoding="utf-8", errors="replace"
        ) as stderr_file:
            result = subprocess.run(
                [sys.executable, str(train_script)],
                cwd=str(ROOT_DIR),
                env=job["env"],
                stdout=stdout_file,
                stderr=stderr_file,
            )
        if result.returncode != 0:
            raise SystemExit(f"PPO-RL sweep failed: {job['tag']} (exit={result.returncode})")

    print("\n[*] PPO-RL baseline sweep finished.")


if __name__ == "__main__":
    main()
