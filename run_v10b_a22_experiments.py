import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_TEACHER_DATASET = ROOT_DIR / "train_log" / "v10b_teacher_dataset_ab_20260429_121552" / "teacher_dataset.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="Run A22: replay old classifier, calibrate, and retrain A2 if needed.")
    parser.add_argument("--teacher-dataset", default=str(DEFAULT_TEACHER_DATASET))
    parser.add_argument("--base-model-tag", default="U220_D230_epoch290")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--eval-full-limit", type=int, default=0)
    parser.add_argument("--eval-hard-limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_plan(args, timestamp: str):
    train_log_dir = ROOT_DIR / "train_log"
    replay_dir = train_log_dir / f"v10b_risk_classifier_replay_{timestamp}"
    calib_a1_dir = train_log_dir / f"v10b_risk_calib_a1_{timestamp}"
    a2_dir = train_log_dir / f"v10b_risk_classifier_a2_{timestamp}"
    calib_a2_dir = train_log_dir / f"v10b_risk_calib_a2_{timestamp}"

    replay_cmd = [
        sys.executable,
        str(ROOT_DIR / "train_v10b_risk_classifier.py"),
        "--teacher-dataset",
        str(Path(args.teacher_dataset).resolve()),
        "--output-dir",
        str(replay_dir),
        "--mode",
        "legacy_replay",
        "--seed",
        str(args.seed),
        "--epochs",
        "20",
        "--eval-full-every",
        "5",
        "--save-epochs",
        "0,5,10,15,20",
    ]
    calib_a1_cmd = [
        sys.executable,
        str(ROOT_DIR / "run_v10b_classifier_calibration.py"),
        "--checkpoint-dir",
        str(replay_dir),
        "--output-dir",
        str(calib_a1_dir),
        "--base-model-tag",
        args.base_model_tag,
    ]
    a2_cmd = [
        sys.executable,
        str(ROOT_DIR / "train_v10b_risk_classifier.py"),
        "--teacher-dataset",
        str(Path(args.teacher_dataset).resolve()),
        "--output-dir",
        str(a2_dir),
        "--mode",
        "a2_binary_critical",
        "--seed",
        str(args.seed),
        "--epochs",
        "20",
        "--eval-full-every",
        "5",
        "--save-epochs",
        "0,5,10,15,20",
    ]
    calib_a2_cmd = [
        sys.executable,
        str(ROOT_DIR / "run_v10b_classifier_calibration.py"),
        "--checkpoint-dir",
        str(a2_dir),
        "--output-dir",
        str(calib_a2_dir),
        "--base-model-tag",
        args.base_model_tag,
    ]

    if int(args.eval_full_limit) > 0:
        replay_cmd.extend(["--eval-full-limit", str(args.eval_full_limit)])
        calib_a1_cmd.extend(["--eval-full-limit", str(args.eval_full_limit)])
        a2_cmd.extend(["--eval-full-limit", str(args.eval_full_limit)])
        calib_a2_cmd.extend(["--eval-full-limit", str(args.eval_full_limit)])
    if int(args.eval_hard_limit) > 0:
        replay_cmd.extend(["--eval-hard-limit", str(args.eval_hard_limit)])
        calib_a1_cmd.extend(["--eval-hard-limit", str(args.eval_hard_limit)])
        a2_cmd.extend(["--eval-hard-limit", str(args.eval_hard_limit)])
        calib_a2_cmd.extend(["--eval-hard-limit", str(args.eval_hard_limit)])

    return {
        "replay_dir": replay_dir,
        "calib_a1_dir": calib_a1_dir,
        "a2_dir": a2_dir,
        "calib_a2_dir": calib_a2_dir,
        "replay_cmd": replay_cmd,
        "calib_a1_cmd": calib_a1_cmd,
        "a2_cmd": a2_cmd,
        "calib_a2_cmd": calib_a2_cmd,
    }


def run_command(name: str, argv: List[str], logs_dir: Path) -> None:
    out_log = logs_dir / f"{name}.out.log"
    err_log = logs_dir / f"{name}.err.log"
    print(f"[*] Running {name}")
    print(f"    stdout: {out_log}")
    print(f"    stderr: {err_log}")
    with out_log.open("w", encoding="utf-8") as out_f, err_log.open("w", encoding="utf-8") as err_f:
        proc = subprocess.Popen(
            argv,
            cwd=str(ROOT_DIR),
            stdout=out_f,
            stderr=err_f,
            text=True,
        )
        return_code = proc.wait()
    if return_code != 0:
        raise RuntimeError(f"Experiment failed: {name} (exit={return_code})")


def load_best_config(calib_dir: Path) -> Dict:
    return json.loads((calib_dir / "best_config.json").read_text(encoding="utf-8"))


def load_baseline_summary(calib_dir: Path) -> Dict:
    return json.loads((calib_dir / "baseline_summary.json").read_text(encoding="utf-8"))


def build_compare_rows(baseline_summary: Dict, old_best: Optional[Dict], a2_best: Optional[Dict]) -> List[Dict]:
    rows = [
        {
            "variant": "baseline_v10b",
            "checkpoint_epoch": "",
            "tau_intervene": "",
            "tau_critical": "",
            "full217_merge_success_rate": baseline_summary["full217"]["merge_success_rate"],
            "full217_endpoint_success_rate": baseline_summary["full217"]["endpoint_success_rate"],
            "full217_safety_success_rate": baseline_summary["full217"]["safety_success_rate"],
            "full217_collision_rate": baseline_summary["full217"]["collision_rate"],
            "full217_shield_rate": baseline_summary["full217"]["shield_intervention_rate"],
            "hard15_merge_success_rate": baseline_summary["hard15"]["merge_success_rate"],
            "hard15_endpoint_success_rate": baseline_summary["hard15"]["endpoint_success_rate"],
            "hard15_safety_success_rate": baseline_summary["hard15"]["safety_success_rate"],
            "hard15_collision_rate": baseline_summary["hard15"]["collision_rate"],
            "hard15_shield_rate": baseline_summary["hard15"]["shield_intervention_rate"],
            "passes_thresholds": True,
        }
    ]
    if old_best:
        row = {"variant": "calibrated_old_a", **old_best}
        rows.append(row)
    if a2_best:
        row = {"variant": "calibrated_a_train_2", **a2_best}
        rows.append(row)
    return rows


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plan = build_plan(args, timestamp)
    manifest_stem = f"v10b_a22_manifest_{timestamp}"
    manifest_path = ROOT_DIR / "train_log" / f"{manifest_stem}.json"
    logs_dir = ROOT_DIR / "train_log" / f"{manifest_stem}_logs"

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "teacher_dataset": str(Path(args.teacher_dataset).resolve()),
        "base_model_tag": args.base_model_tag,
        "seed": int(args.seed),
        "logs_dir": str(logs_dir),
        "plan": {
            "replay_dir": str(plan["replay_dir"]),
            "calib_a1_dir": str(plan["calib_a1_dir"]),
            "a2_dir": str(plan["a2_dir"]),
            "calib_a2_dir": str(plan["calib_a2_dir"]),
            "replay_cmd": plan["replay_cmd"],
            "calib_a1_cmd": plan["calib_a1_cmd"],
            "a2_cmd": plan["a2_cmd"],
            "calib_a2_cmd": plan["calib_a2_cmd"],
        },
    }
    save_json(manifest_path, manifest)

    print("=" * 80)
    print("v10b A22 experiment plan")
    print(f"Manifest: {manifest_path}")
    print(f"Logs dir: {logs_dir}")
    print(f"1. replay old A -> {plan['replay_dir']}")
    print(f"2. A-Calib-1 -> {plan['calib_a1_dir']}")
    print(f"3. A-Train-2 -> {plan['a2_dir']} (only if A-Calib-1 fails)")
    print(f"4. A2 calibration -> {plan['calib_a2_dir']}")
    print("=" * 80)

    if args.dry_run:
        print("[Dry run] No subprocess started.")
        return

    logs_dir.mkdir(parents=True, exist_ok=True)
    run_command("a22_replay_old_a", plan["replay_cmd"], logs_dir)
    run_command("a22_calib_a1", plan["calib_a1_cmd"], logs_dir)

    calib_a1_best = load_best_config(plan["calib_a1_dir"])
    baseline_summary = load_baseline_summary(plan["calib_a1_dir"])
    stop_after_a1 = bool(calib_a1_best.get("passes_thresholds", False))
    calib_a2_best = None

    if stop_after_a1:
        print("[*] A-Calib-1 already passed thresholds. Skip A-Train-2.")
    else:
        run_command("a22_train_a2", plan["a2_cmd"], logs_dir)
        run_command("a22_calib_a2", plan["calib_a2_cmd"], logs_dir)
        calib_a2_best = load_best_config(plan["calib_a2_dir"])

    compare_rows = build_compare_rows(
        baseline_summary=baseline_summary,
        old_best=calib_a1_best.get("best_row"),
        a2_best=calib_a2_best.get("best_row") if calib_a2_best else None,
    )
    save_csv(ROOT_DIR / "train_log" / f"{manifest_stem}_compare.csv", compare_rows)
    save_json(
        ROOT_DIR / "train_log" / f"{manifest_stem}_compare.json",
        {
            "baseline_summary": baseline_summary,
            "old_a_best": calib_a1_best,
            "a2_best": calib_a2_best,
            "stopped_after_a1": stop_after_a1,
        },
    )
    print("[*] v10b A22 experiments finished.")


if __name__ == "__main__":
    main()
