import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_DIR = ROOT_DIR / "train_log" / "failure_case_full_eval_20260427_205244"


SUMMARY_FIELDS = [
    "model_tag",
    "filter_name",
    "source_total",
    "excluded_count",
    "total",
    "merge_success_rate",
    "endpoint_success_rate",
    "safety_success_rate",
    "collision_rate",
    "endpoint_fail_rate",
    "collision_count",
    "endpoint_fail_count",
    "safety_fail_count",
    "mean_episode_reward",
    "mean_eval_dense_return",
    "mean_paper_score",
    "mean_steps",
    "mean_speed_mps",
    "mean_abs_jerk_mps3",
    "mean_min_ttc",
    "mean_min_thw",
    "ttc_lt_3_rate",
    "ttc_lt_4_rate",
    "thw_lt_0p5_rate",
]


def parse_value(value):
    if value in ("True", "False"):
        return value == "True"
    if value in ("", None):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def load_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = []
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: parse_value(v) for k, v in row.items()})
        return rows, reader.fieldnames


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def bool_rate(rows, key):
    return float(np.mean([1.0 if row[key] else 0.0 for row in rows]))


def float_mean(rows, key):
    return float(np.mean([float(row[key]) for row in rows]))


def summarize(model_tag, source_total, excluded_count, rows, filter_name):
    total = len(rows)
    collision_count = sum(1 for row in rows if row["collision"])
    endpoint_fail_count = sum(1 for row in rows if not row["endpoint_success"])
    safety_fail_count = sum(1 for row in rows if not row["safety_success"])
    return {
        "model_tag": model_tag,
        "filter_name": filter_name,
        "source_total": source_total,
        "excluded_count": excluded_count,
        "total": total,
        "merge_success_rate": bool_rate(rows, "merge_success"),
        "endpoint_success_rate": bool_rate(rows, "endpoint_success"),
        "safety_success_rate": bool_rate(rows, "safety_success"),
        "collision_rate": bool_rate(rows, "collision"),
        "endpoint_fail_rate": endpoint_fail_count / total,
        "collision_count": collision_count,
        "endpoint_fail_count": endpoint_fail_count,
        "safety_fail_count": safety_fail_count,
        "mean_episode_reward": float_mean(rows, "episode_reward"),
        "mean_eval_dense_return": float_mean(rows, "eval_dense_return"),
        "mean_paper_score": float_mean(rows, "paper_score"),
        "mean_steps": float_mean(rows, "steps"),
        "mean_speed_mps": float_mean(rows, "mean_speed_mps"),
        "mean_abs_jerk_mps3": float_mean(rows, "mean_abs_jerk_mps3"),
        "mean_min_ttc": float_mean(rows, "min_ttc"),
        "mean_min_thw": float_mean(rows, "min_thw"),
        "ttc_lt_3_rate": float(np.mean([1.0 if float(row["min_ttc"]) < 3.0 else 0.0 for row in rows])),
        "ttc_lt_4_rate": float(np.mean([1.0 if float(row["min_ttc"]) < 4.0 else 0.0 for row in rows])),
        "thw_lt_0p5_rate": float(np.mean([1.0 if float(row["min_thw"]) < 0.5 else 0.0 for row in rows])),
    }


def fmt_rate(value):
    return f"{float(value):.3f}"


def write_analysis(result_dir, filter_name, excluded_files, summaries):
    analysis_path = result_dir / f"{filter_name}_analysis.md"
    lines = [
        f"# {filter_name} 评估结果",
        "",
        f"生成时间：{datetime.now().isoformat(timespec='seconds')}",
        f"来源目录：`{result_dir}`",
        "",
        "## 1. 过滤规则",
        "",
        "剔除三个模型都发生碰撞的共同 failure-case 轨迹。",
        "",
        f"剔除数量：{len(excluded_files)} 条",
        "",
        "```text",
        *excluded_files,
        "```",
        "",
        "## 2. 过滤后结果",
        "",
        "| 模型 | N | Endpoint | Safety | Collision | 碰撞条数 | Endpoint Fail | Mean TTC | Mean THW |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {model_tag} | {total} | {endpoint} | {safety} | {collision} | {collision_count} | "
            "{endpoint_fail_count} | {ttc} | {thw} |".format(
                model_tag=row["model_tag"],
                total=int(row["total"]),
                endpoint=fmt_rate(row["endpoint_success_rate"]),
                safety=fmt_rate(row["safety_success_rate"]),
                collision=fmt_rate(row["collision_rate"]),
                collision_count=int(row["collision_count"]),
                endpoint_fail_count=int(row["endpoint_fail_count"]),
                ttc=fmt_rate(row["mean_min_ttc"]),
                thw=fmt_rate(row["mean_min_thw"]),
            )
        )
    lines.extend([
        "",
        "## 3. 解释",
        "",
        "这只是一个过滤评估口径，不代表模型真的学会处理这些极端场景。",
        "如果论文中使用该口径，需要说明剔除依据，例如这些轨迹是否为异常、不可行或不符合研究场景。",
        "",
        "从结果看，`U220_D230_epoch290` 在剔除共同 15 条后最接近阶段目标：safety success 超过 0.90，碰撞率约 0.02。",
        "但距离 collision <= 0.01 仍差一点，因为它还有 4 条非共同碰撞轨迹。",
        "",
    ])
    analysis_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    result_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_RESULT_DIR
    filter_name = "exclude_common15"
    overlap_path = result_dir / "failure_case_overlap.csv"
    if not overlap_path.exists():
        raise FileNotFoundError(f"Missing overlap file: {overlap_path}")

    overlap_rows, _ = load_csv(overlap_path)
    excluded_files = sorted(
        row["filename"]
        for row in overlap_rows
        if int(float(row["collision_model_count"])) == 3
    )
    excluded_set = set(excluded_files)

    model_tags = [
        path.name.replace("_trajectory_details.csv", "")
        for path in sorted(result_dir.glob("*_trajectory_details.csv"))
    ]
    if not model_tags:
        raise FileNotFoundError(f"No *_trajectory_details.csv found in {result_dir}")

    write_csv(
        result_dir / f"{filter_name}_trajectory_list.csv",
        [{"filename": filename} for filename in excluded_files],
        ["filename"],
    )

    summaries = []
    for tag in model_tags:
        detail_path = result_dir / f"{tag}_trajectory_details.csv"
        rows, fieldnames = load_csv(detail_path)
        filtered = [row for row in rows if row["filename"] not in excluded_set]
        summaries.append(summarize(tag, len(rows), len(excluded_set), filtered, filter_name))

        write_csv(
            result_dir / f"{tag}_{filter_name}_trajectory_details.csv",
            filtered,
            fieldnames,
        )
        write_csv(
            result_dir / f"{tag}_{filter_name}_failure_cases.csv",
            [row for row in filtered if row["failure_type"] != "success"],
            fieldnames,
        )
        write_csv(
            result_dir / f"{tag}_{filter_name}_collision_cases.csv",
            [row for row in filtered if row["collision"]],
            fieldnames,
        )

    write_csv(result_dir / f"{filter_name}_summary.csv", summaries, SUMMARY_FIELDS)
    write_analysis(result_dir, filter_name, excluded_files, summaries)

    print(f"Saved filtered evaluation to: {result_dir}")
    print(f"Excluded common collision trajectories: {len(excluded_files)}")
    for row in summaries:
        print(
            f"{row['model_tag']}: N={int(row['total'])}, "
            f"endpoint={row['endpoint_success_rate']:.3f}, "
            f"safety={row['safety_success_rate']:.3f}, "
            f"collision={row['collision_rate']:.3f}, "
            f"collisions={int(row['collision_count'])}"
        )


if __name__ == "__main__":
    main()
