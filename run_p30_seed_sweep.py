import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
STEPS_PER_EPOCH = 2048
U220_TIMESTEPS = 220 * STEPS_PER_EPOCH


P30_ENV = {
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
    "PROBE_PPO_EPOCHS": "6",
    "PROBE_PPO_MINI_BATCH_SIZE": "256",
    "PROBE_GENERATOR_LR": "8e-5",
    "PROBE_DISCRIMINATOR_LR": "5e-5",
    "PROBE_N_DISC_UPDATES": "5",
    "PROBE_ENT_COEF": "0.005",
    "PROBE_SAFETY_UNFREEZE_TIMESTEPS": str(U220_TIMESTEPS),
    "PROBE_SAFETY_LIGHT_UNFREEZE_LR": "5e-6",
    "PROBE_LATE_N_DISC_EPOCH": "230",
    "PROBE_LATE_N_DISC_UPDATES": "4",
    "PROBE_SAFETY_DECAY_EPOCH": "250",
    "PROBE_SAFETY_DECAY_LR": "2.5e-6",
    "PROBE_SAFETY_RAMP_UNFREEZE_EPOCHS": "0",
    "PROBE_SAFETY_DECAY_RAMP_EPOCHS": "0",
    "PROBE_ENABLE_SAFETY_MODULE": "1",
    "PROBE_ENABLE_SAFETY_BRANCH": "1",
    "PROBE_ENABLE_SAFETY_AUX_LOSS": "1",
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


def parse_args():
    parser = argparse.ArgumentParser(description="Run P30 seed sweep sequentially.")
    parser.add_argument("--seeds", default="45,46,47,48,49", help="Comma-separated seeds.")
    parser.add_argument("--wait-pid", type=int, default=0, help="Wait for an existing training process before starting.")
    parser.add_argument("--output-dir", default="", help="Optional launcher log directory.")
    parser.add_argument("--full-eval-episodes", default="", help="Override PROBE_FULL_EVAL_EPISODES.")
    parser.add_argument("--epoch0-eval-episodes", default="", help="Override PROBE_EPOCH0_EVAL_EPISODES.")
    parser.add_argument("--tag-prefix", default="P30_CPairD250_NoLateLR_Save1", help="PROBE_TAG prefix before _seedN.")
    return parser.parse_args()


def process_exists(pid):
    if pid <= 0:
        return False
    result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-Command",
            f"if (Get-Process -Id {pid} -ErrorAction SilentlyContinue) {{ exit 0 }} else {{ exit 1 }}",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def write_line(path, text):
    with path.open("a", encoding="utf-8") as fp:
        fp.write(text + "\n")


def main():
    args = parse_args()
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "train_log" / f"p30_seed_sweep_{seeds[0]}_{seeds[-1]}_{stamp}_logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    runner_log = output_dir / "runner.log"

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python": sys.executable,
        "seeds": seeds,
        "wait_pid": args.wait_pid,
        "base_env": P30_ENV,
        "full_eval_episodes_override": args.full_eval_episodes,
        "epoch0_eval_episodes_override": args.epoch0_eval_episodes,
        "tag_prefix": args.tag_prefix,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    write_line(runner_log, f"QUEUED seeds={seeds} wait_pid={args.wait_pid} at {datetime.now().isoformat(timespec='seconds')}")
    if args.wait_pid and process_exists(args.wait_pid):
        write_line(runner_log, f"WAIT pid={args.wait_pid} at {datetime.now().isoformat(timespec='seconds')}")
        while process_exists(args.wait_pid):
            time.sleep(30)
        write_line(runner_log, f"WAIT_DONE pid={args.wait_pid} at {datetime.now().isoformat(timespec='seconds')}")

    for seed in seeds:
        env = os.environ.copy()
        run_env = dict(P30_ENV)
        if args.full_eval_episodes:
            run_env["PROBE_FULL_EVAL_EPISODES"] = str(args.full_eval_episodes)
        if args.epoch0_eval_episodes:
            run_env["PROBE_EPOCH0_EVAL_EPISODES"] = str(args.epoch0_eval_episodes)
        env.update(run_env)
        env["PROBE_SEED"] = str(seed)
        env["PROBE_TAG"] = f"{args.tag_prefix}_seed{seed}"
        env["PYTHONUNBUFFERED"] = "1"
        stdout_path = output_dir / f"seed_{seed}.out.log"
        stderr_path = output_dir / f"seed_{seed}.err.log"

        write_line(runner_log, f"START seed={seed} at {datetime.now().isoformat(timespec='seconds')}")
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
        write_line(runner_log, f"END seed={seed} exit={result.returncode} at {datetime.now().isoformat(timespec='seconds')}")
        if result.returncode != 0:
            raise SystemExit(result.returncode)

    write_line(runner_log, f"FINISHED all seeds at {datetime.now().isoformat(timespec='seconds')}")


if __name__ == "__main__":
    main()
