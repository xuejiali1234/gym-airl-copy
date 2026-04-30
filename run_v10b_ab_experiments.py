import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run v10b A/B experiments: teacher v2 export + risk classifier + residual corrector."
    )
    parser.add_argument("--base-model-tag", default="U220_D230_epoch290")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--teacher-normal-safe-keep-prob", type=float, default=1.0)
    parser.add_argument("--teacher-limit-trajectories", type=int, default=0)
    parser.add_argument("--classifier-epochs", type=int, default=20)
    parser.add_argument("--residual-epochs", type=int, default=20)
    parser.add_argument("--eval-full-every", type=int, default=5)
    parser.add_argument("--eval-full-limit", type=int, default=0)
    parser.add_argument("--eval-hard-limit", type=int, default=0)
    parser.add_argument("--classifier-threshold", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def save_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_command_plan(args, timestamp: str):
    train_log_dir = ROOT_DIR / "train_log"
    teacher_dir = train_log_dir / f"v10b_teacher_dataset_ab_{timestamp}"
    classifier_dir = train_log_dir / f"v10b_risk_classifier_{timestamp}"
    residual_dir = train_log_dir / f"v10b_residual_corrector_{timestamp}"
    teacher_dataset = teacher_dir / "teacher_dataset.pt"

    common_eval_args = [
        "--eval-full-every",
        str(args.eval_full_every),
        "--eval-full-limit",
        str(args.eval_full_limit),
        "--eval-hard-limit",
        str(args.eval_hard_limit),
    ]

    export_argv = [
        sys.executable,
        str(ROOT_DIR / "export_v10b_teacher_dataset.py"),
        "--base-model-tag",
        args.base_model_tag,
        "--output-dir",
        str(teacher_dir),
        "--seed",
        str(args.seed),
        "--normal-safe-keep-prob",
        str(args.teacher_normal_safe_keep_prob),
    ]
    if int(args.teacher_limit_trajectories) > 0:
        export_argv.extend(["--limit-trajectories", str(args.teacher_limit_trajectories)])

    commands = [
        {
            "name": "teacher_export_ab_v2",
            "output_dir": str(teacher_dir),
            "argv": export_argv,
        },
        {
            "name": "planA_risk_classifier",
            "output_dir": str(classifier_dir),
            "argv": [
                sys.executable,
                str(ROOT_DIR / "train_v10b_risk_classifier.py"),
                "--teacher-dataset",
                str(teacher_dataset),
                "--output-dir",
                str(classifier_dir),
                "--seed",
                str(args.seed),
                "--epochs",
                str(args.classifier_epochs),
                "--threshold",
                str(args.classifier_threshold),
                *common_eval_args,
            ],
        },
        {
            "name": "planB_residual_corrector",
            "output_dir": str(residual_dir),
            "argv": [
                sys.executable,
                str(ROOT_DIR / "train_v10b_residual_corrector.py"),
                "--teacher-dataset",
                str(teacher_dataset),
                "--output-dir",
                str(residual_dir),
                "--seed",
                str(args.seed),
                "--epochs",
                str(args.residual_epochs),
                *common_eval_args,
            ],
        },
    ]
    return teacher_dir, commands


def run_command(entry: dict, logs_dir: Path) -> None:
    out_log = logs_dir / f"{entry['name']}.out.log"
    err_log = logs_dir / f"{entry['name']}.err.log"
    print(f"[*] Running {entry['name']}")
    print(f"    stdout: {out_log}")
    print(f"    stderr: {err_log}")
    with out_log.open("w", encoding="utf-8") as out_f, err_log.open("w", encoding="utf-8") as err_f:
        proc = subprocess.Popen(
            entry["argv"],
            cwd=str(ROOT_DIR),
            stdout=out_f,
            stderr=err_f,
            text=True,
        )
        return_code = proc.wait()
    if return_code != 0:
        raise RuntimeError(f"Experiment failed: {entry['name']} (exit={return_code})")


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    teacher_dir, commands = build_command_plan(args, timestamp)
    manifest_stem = f"v10b_ab_manifest_{timestamp}"
    manifest_path = ROOT_DIR / "train_log" / f"{manifest_stem}.json"
    logs_dir = ROOT_DIR / "train_log" / f"{manifest_stem}_logs"

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "base_model_tag": args.base_model_tag,
        "seed": int(args.seed),
        "teacher_output_dir": str(teacher_dir),
        "logs_dir": str(logs_dir),
        "commands": [
            {
                "name": entry["name"],
                "output_dir": entry["output_dir"],
                "argv": entry["argv"],
            }
            for entry in commands
        ],
    }
    save_manifest(manifest_path, manifest)

    print("=" * 80)
    print("v10b A/B experiment plan")
    print(f"Manifest: {manifest_path}")
    print(f"Logs dir: {logs_dir}")
    for idx, entry in enumerate(commands, start=1):
        print(f"{idx}. {entry['name']}: {entry['output_dir']}")
        print(f"   argv={entry['argv']}")
    print("=" * 80)

    if args.dry_run:
        print("[Dry run] No subprocess started.")
        return

    logs_dir.mkdir(parents=True, exist_ok=True)
    for entry in commands:
        run_command(entry, logs_dir)
    print("[*] v10b A/B experiments finished.")


if __name__ == "__main__":
    main()
