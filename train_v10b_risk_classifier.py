import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

from configs.config import Config
from envs.merging_env import MergingEnv
from evaluation.failure_case_full_evaluate import SingleTrajDataset
from evaluation.safety_shield_evaluate import action_risk, shield_action, target_lane_vehicle_metrics
from model.attention_net import AttentionFeaturesExtractor, GoalConditionedMLPFeaturesExtractor  # noqa: F401
from model.safety_oracle_q import SafetyOracleQ
from v10b_ab_common import (
    RISK_REASON_LABELS,
    RISK_SIDE_LABELS,
    append_csv_row,
    binary_precision,
    binary_recall,
    build_ab_input_matrix,
    make_classification_arrays,
    multiclass_accuracy,
    sanitize_numeric,
    teacher_split_indices,
)
from v10b_distill_common import (
    BASE_MODEL_TAG,
    build_v10b_shield_params,
    get_model_info_by_tag,
    load_eval_dataset,
    load_teacher_payload,
    save_json as save_json_common,
    seed_everything,
)


ROOT_DIR = Path(__file__).resolve().parent
LEGACY_MODE = "legacy_replay"
A2_MODE = "a2_binary_critical"
A3_MODE = "a3_conservative_gate"
DEFAULT_SAVE_EPOCHS = (0, 5, 10, 15, 20)


class V10BRiskClassifier(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.intervene_head = nn.Linear(128, 1)
        self.risk_side_head = nn.Linear(128, len(RISK_SIDE_LABELS))
        self.risk_reason_head = nn.Linear(128, len(RISK_REASON_LABELS))
        self.critical_head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.encoder(x)
        return {
            "intervene_logit": self.intervene_head(feat).squeeze(-1),
            "risk_side_logits": self.risk_side_head(feat),
            "risk_reason_logits": self.risk_reason_head(feat),
            "critical_logit": self.critical_head(feat).squeeze(-1),
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Train v10b risk classifiers without touching AIRL.")
    parser.add_argument("--teacher-dataset", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--mode", choices=[LEGACY_MODE, A2_MODE, A3_MODE], default=LEGACY_MODE)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval-full-every", type=int, default=5)
    parser.add_argument("--eval-full-limit", type=int, default=0)
    parser.add_argument("--eval-hard-limit", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--save-epochs", default="0,5,10,15,20")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_epoch_list(raw: str) -> Tuple[int, ...]:
    epochs = sorted({int(item.strip()) for item in str(raw).split(",") if item.strip()})
    return tuple(epochs)


def default_output_dir(mode: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if mode == LEGACY_MODE:
        stem = "v10b_risk_classifier"
    elif mode == A2_MODE:
        stem = "v10b_risk_classifier_a2"
    else:
        stem = "v10b_risk_classifier_a3"
    return ROOT_DIR / "train_log" / f"{stem}_{timestamp}"


def build_oracle_baseline(cfg: Config, dataset, samples: Sequence[Dict]) -> Dict[str, np.ndarray]:
    obs = np.stack([sanitize_numeric(np.asarray(item["obs"], dtype=np.float32)) for item in samples], axis=0)
    acts = np.stack(
        [sanitize_numeric(np.asarray(item["policy_action"], dtype=np.float32), posinf=1.0, neginf=-1.0) for item in samples],
        axis=0,
    )
    device = torch.device("cpu")
    oracle = SafetyOracleQ(cfg, dataset.expert_mean, dataset.expert_std)
    with torch.no_grad():
        labels = oracle.check_safety_batch(
            torch.as_tensor(obs, device=device),
            torch.as_tensor(acts, device=device),
        ).cpu().numpy().reshape(-1)
    pred = (labels >= 0.5).astype(np.float32)
    return {"intervene_pred": pred}


def build_training_targets(mode: str, arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    hard_multiplier = np.where(arrays["hard15"], 5.0, 1.0).astype(np.float32)
    intervene_positive = arrays["intervene"] > 0.5
    warning_only = arrays["warning"] & ~intervene_positive

    if mode == LEGACY_MODE:
        common_weight = np.full(len(arrays["intervene"]), 0.5, dtype=np.float32)
        common_weight[arrays["warning"]] = 1.0
        common_weight[intervene_positive] = 3.0
        common_weight *= hard_multiplier
        return {
            "intervene_target": arrays["intervene"].astype(np.float32),
            "critical_target": arrays["critical"].astype(np.float32),
            "intervene_weight": common_weight,
            "critical_weight": common_weight,
            "side_weight": common_weight,
            "reason_weight": common_weight,
        }

    if mode == A3_MODE:
        return {
            "intervene_target": np.where(intervene_positive, 1.0, np.where(warning_only, 0.2, 0.0)).astype(np.float32),
            "critical_target": arrays["critical"].astype(np.float32),
            "intervene_weight": hard_multiplier
            * np.where(intervene_positive, 3.0, np.where(warning_only, 1.0, 1.5)).astype(np.float32),
            "critical_weight": hard_multiplier * np.where(arrays["critical"] > 0.5, 6.0, 1.5).astype(np.float32),
            "side_weight": hard_multiplier,
            "reason_weight": hard_multiplier,
        }

    return {
        "intervene_target": np.where(intervene_positive, 1.0, np.where(warning_only, 0.4, 0.0)).astype(np.float32),
        "critical_target": arrays["critical"].astype(np.float32),
        "intervene_weight": hard_multiplier
        * np.where(intervene_positive, 3.0, np.where(warning_only, 1.5, 1.0)).astype(np.float32),
        "critical_weight": hard_multiplier * np.where(arrays["critical"] > 0.5, 10.0, 1.0).astype(np.float32),
        "side_weight": hard_multiplier,
        "reason_weight": hard_multiplier,
    }


def weighted_mean(loss_per: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (loss_per * weights).sum() / weights.sum().clamp_min(1e-6)


def compute_offline_metrics(
    model: V10BRiskClassifier,
    inputs: np.ndarray,
    arrays: Dict[str, np.ndarray],
    split_indices: Dict[str, np.ndarray],
    threshold_intervene: float,
    threshold_critical: Optional[float] = None,
) -> Dict[str, float]:
    model.eval()
    model_device = next(model.parameters()).device
    threshold_critical = threshold_intervene if threshold_critical is None else threshold_critical
    with torch.no_grad():
        logits = model(torch.as_tensor(inputs, dtype=torch.float32, device=model_device))
        intervene_prob = torch.sigmoid(logits["intervene_logit"]).cpu().numpy()
        critical_prob = torch.sigmoid(logits["critical_logit"]).cpu().numpy()
        risk_side_pred = logits["risk_side_logits"].argmax(dim=1).cpu().numpy()
        risk_reason_pred = logits["risk_reason_logits"].argmax(dim=1).cpu().numpy()

    intervene_pred = (intervene_prob >= threshold_intervene).astype(np.float32)
    critical_pred = (critical_prob >= threshold_critical).astype(np.float32)

    val_idx = split_indices["val"]
    hard_idx = split_indices["hard"]
    full_idx = split_indices["full"]

    metrics = {
        "hard15_intervene_recall": binary_recall(arrays["intervene"][hard_idx], intervene_pred[hard_idx]),
        "hard15_critical_recall": binary_recall(arrays["critical"][hard_idx], critical_pred[hard_idx]),
        "full217_intervene_recall": binary_recall(arrays["intervene"][full_idx], intervene_pred[full_idx]),
        "full217_intervene_precision": binary_precision(arrays["intervene"][full_idx], intervene_pred[full_idx]),
        "val_intervene_recall": binary_recall(arrays["intervene"][val_idx], intervene_pred[val_idx]) if len(val_idx) else 0.0,
        "val_intervene_precision": binary_precision(arrays["intervene"][val_idx], intervene_pred[val_idx]) if len(val_idx) else 0.0,
        "val_risk_side_acc": multiclass_accuracy(arrays["risk_side"][val_idx], risk_side_pred[val_idx]) if len(val_idx) else 0.0,
        "val_risk_reason_acc": multiclass_accuracy(arrays["risk_reason"][val_idx], risk_reason_pred[val_idx]) if len(val_idx) else 0.0,
        "val_critical_recall": binary_recall(arrays["critical"][val_idx], critical_pred[val_idx]) if len(val_idx) else 0.0,
    }
    return metrics


def evaluate_classifier_protocol(
    base_model: PPO,
    classifier: V10BRiskClassifier,
    dataset,
    cfg: Config,
    traj_indices: Sequence[int],
    shield_params: Dict,
    tau_intervene: float,
    tau_critical: float,
    device: torch.device,
    *,
    critical_overrides_gate: bool,
    use_raw_critical_rule: bool,
    raw_critical_override_mode: str = "all",
) -> Dict[str, float]:
    total = 0
    merge_success = 0
    endpoint_success = 0
    safety_success = 0
    collision_count = 0
    predicted_interventions = 0
    teacher_interventions = 0
    false_negatives = 0
    critical_count = 0
    critical_false_negatives = 0
    raw_critical_override_count = 0
    total_steps = 0

    def effective_raw_critical(score: Dict[str, float]) -> bool:
        raw_critical = bool(score.get("critical", False))
        if not use_raw_critical_rule or not raw_critical:
            return False
        if raw_critical_override_mode == "all":
            return True
        if raw_critical_override_mode == "overlap_or_ttc":
            if bool(score.get("collided", False)) or bool(score.get("predicted_overlap", False)):
                return True
            return float(score.get("min_ttc", 20.0)) < float(shield_params.get("critical_ttc_min", 1.0))
        if raw_critical_override_mode == "overlap_only":
            return bool(score.get("collided", False)) or bool(score.get("predicted_overlap", False))
        if raw_critical_override_mode == "none":
            return False
        raise ValueError(f"Unsupported raw_critical_override_mode: {raw_critical_override_mode}")

    for traj_index in tqdm(traj_indices, desc="risk_classifier_eval", leave=False):
        traj = dataset[traj_index]
        env = MergingEnv(SingleTrajDataset(traj, dataset.expert_mean, dataset.expert_std))
        env.collision_margin = 1.0
        obs, _ = env.reset(seed=cfg.SEED + int(traj_index))
        terminated = False
        truncated = False
        max_steps = len(env.current_traj["ego_pos"]) + 50
        step_count = 0
        teacher_shield_interventions = 0
        teacher_consecutive = 0
        teacher_safe_steps = 0
        last_info = {}

        while not (terminated or truncated) and step_count < max_steps:
            policy_action, _ = base_model.predict(obs, deterministic=True)
            policy_action = np.asarray(policy_action, dtype=np.float32).reshape(-1)
            original_score = action_risk(env, policy_action, policy_action, shield_params)
            raw_critical = bool(original_score.get("critical", False))
            effective_override_critical = effective_raw_critical(original_score)
            pair = target_lane_vehicle_metrics(env)
            aux = np.asarray(
                [
                    pair["lead"]["gap"],
                    pair["lead"]["thw"],
                    pair["follow"]["gap"],
                    pair["follow"]["thw"],
                ],
                dtype=np.float32,
            )
            x = np.concatenate(
                [sanitize_numeric(np.asarray(obs, dtype=np.float32)), sanitize_numeric(policy_action), sanitize_numeric(aux)],
                axis=0,
            )[None, :]
            classifier.eval()
            with torch.no_grad():
                pred = classifier(torch.as_tensor(x, dtype=torch.float32, device=device))
                intervene_prob = float(torch.sigmoid(pred["intervene_logit"]).item())
                critical_prob = float(torch.sigmoid(pred["critical_logit"]).item())
            intervene_gate = intervene_prob >= tau_intervene
            critical_pred = (critical_prob >= tau_critical) or effective_override_critical
            call_shield = intervene_gate or (critical_overrides_gate and critical_pred)

            teacher_action, teacher_info = shield_action(
                env=env,
                policy_action=policy_action,
                params=shield_params,
                step_count=step_count,
                shield_interventions=teacher_shield_interventions,
                consecutive_interventions=teacher_consecutive,
                safe_steps=teacher_safe_steps,
            )
            teacher_action = np.asarray(teacher_action, dtype=np.float32).reshape(-1)
            teacher_intervene = bool(teacher_info["intervened"])
            teacher_critical = raw_critical
            if effective_override_critical:
                raw_critical_override_count += 1

            if call_shield:
                action = teacher_action
                predicted_interventions += 1
            else:
                action = policy_action
            if teacher_intervene:
                teacher_interventions += 1
                teacher_shield_interventions += 1
                teacher_consecutive += 1
                if not call_shield:
                    false_negatives += 1
            else:
                teacher_consecutive = 0
            if teacher_critical:
                critical_count += 1
                if not critical_pred:
                    critical_false_negatives += 1

            obs, _, terminated, truncated, info = env.step(action)
            last_info = dict(info)
            total_steps += 1

            if (
                not bool(getattr(env, "has_collided_this_episode", False))
                and hasattr(env, "_get_surround_at_t")
            ):
                surr_now = env._get_surround_at_t(env.t)
                px, py, _, vy = [float(x) for x in env.ego_state]
                min_ttc, min_thw = env._compute_min_ttc_thw(px, py, vy, surr_now)
                if (
                    min_ttc >= shield_params.get("lead_ttc_min", 3.0)
                    and min_thw >= shield_params.get("follow_thw_min", 0.8)
                ):
                    teacher_safe_steps += 1
                else:
                    teacher_safe_steps = 0
            else:
                teacher_safe_steps = 0
            step_count += 1

        total += 1
        merge_success += int(bool(last_info.get("is_merge_success", False)))
        endpoint_success += int(bool(last_info.get("is_endpoint_success", False)))
        safety_success += int(bool(last_info.get("is_safety_success", False)))
        collision_count += int(bool(last_info.get("is_collided", False) or getattr(env, "has_collided_this_episode", False)))

    return {
        "total": total,
        "merge_success_rate": merge_success / max(total, 1),
        "endpoint_success_rate": endpoint_success / max(total, 1),
        "safety_success_rate": safety_success / max(total, 1),
        "collision_rate": collision_count / max(total, 1),
        "collision_count": collision_count,
        "predicted_intervention_rate": predicted_interventions / max(total_steps, 1),
        "teacher_intervention_rate": teacher_interventions / max(total_steps, 1),
        "false_negative_rate": false_negatives / max(teacher_interventions, 1),
        "critical_false_negative_rate": critical_false_negatives / max(critical_count, 1),
        "raw_critical_override_rate": raw_critical_override_count / max(total_steps, 1),
    }


def best_key(offline: Dict[str, float], protocol_full: Dict[str, float], protocol_hard: Dict[str, float]) -> Tuple[float, float, float, float, float]:
    return (
        float(offline["hard15_intervene_recall"]),
        float(offline["hard15_critical_recall"]),
        -float(protocol_full["false_negative_rate"]),
        float(offline["full217_intervene_precision"]),
        -float(protocol_full["predicted_intervention_rate"]),
    )


def clear_output_paths(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in (
        output_dir / "loss.csv",
        output_dir / "eval_summary.csv",
        output_dir / "best_checkpoint.pt",
        output_dir / "best_model_info.json",
        output_dir / "run_summary.json",
    ):
        if stale_path.exists():
            stale_path.unlink()
    for stale_ckpt in output_dir.glob("epoch_*.pt"):
        stale_ckpt.unlink()


def save_epoch_checkpoint(output_dir: Path, classifier: V10BRiskClassifier, epoch: int) -> Path:
    checkpoint_path = output_dir / f"epoch_{epoch}.pt"
    with checkpoint_path.open("wb") as f:
        torch.save({"state_dict": classifier.state_dict(), "epoch": int(epoch)}, f)
    return checkpoint_path


def main():
    args = parse_args()
    save_epochs = parse_epoch_list(args.save_epochs)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else default_output_dir(args.mode).resolve()
    teacher_payload = load_teacher_payload(Path(args.teacher_dataset).resolve())
    samples = teacher_payload["samples"]
    seed_everything(args.seed, deterministic=True)

    if args.dry_run:
        print("=" * 80)
        print("v10b risk classifier dry-run")
        print(f"mode={args.mode}")
        print(f"teacher_dataset={Path(args.teacher_dataset).resolve()}")
        print(f"output_dir={output_dir}")
        print(f"samples={len(samples)}")
        print(f"save_epochs={save_epochs}")
        print("=" * 80)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()
    shield_params = build_v10b_shield_params()
    eval_dataset = load_eval_dataset(device=device)
    base_model = PPO.load(str(get_model_info_by_tag(BASE_MODEL_TAG)["checkpoint"]), device=device)

    inputs = build_ab_input_matrix(samples)
    arrays = make_classification_arrays(samples)
    targets = build_training_targets(args.mode, arrays)
    splits = teacher_split_indices(samples, seed=args.seed)

    classifier = V10BRiskClassifier(input_dim=inputs.shape[1]).to(device)
    optimizer = Adam(classifier.parameters(), lr=float(args.lr))

    clear_output_paths(output_dir)
    save_json_common(
        output_dir / "config.json",
        {
            "teacher_dataset": str(Path(args.teacher_dataset).resolve()),
            "mode": args.mode,
            "seed": int(args.seed),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "threshold": float(args.threshold),
            "save_epochs": list(save_epochs),
            "train_samples": int(len(splits["train"])),
            "val_samples": int(len(splits["val"])),
            "hard_samples": int(len(splits["hard"])),
        },
    )

    oracle_baseline = build_oracle_baseline(cfg, eval_dataset, samples)
    oracle_fn_full = 1.0 - binary_recall(arrays["intervene"][splits["full"]], oracle_baseline["intervene_pred"][splits["full"]])
    oracle_fn_hard = 1.0 - binary_recall(arrays["intervene"][splits["hard"]], oracle_baseline["intervene_pred"][splits["hard"]])

    loss_csv = output_dir / "loss.csv"
    eval_csv = output_dir / "eval_summary.csv"
    best = None
    best_epoch = None
    saved_checkpoint_paths: Dict[str, str] = {}
    hard_indices = sorted(set(int(samples[idx]["traj_index"]) for idx in splits["hard"]))
    full_indices = list(range(len(eval_dataset)))
    if args.eval_hard_limit:
        hard_indices = hard_indices[: int(args.eval_hard_limit)]
    if args.eval_full_limit:
        full_indices = full_indices[: int(args.eval_full_limit)]

    for epoch in range(0, int(args.epochs) + 1):
        if epoch > 0:
            classifier.train()
            perm = np.random.permutation(splits["train"])
            epoch_loss = 0.0
            epoch_intervene = 0.0
            epoch_side = 0.0
            epoch_reason = 0.0
            epoch_critical = 0.0
            batch_count = 0

            for start in range(0, len(perm), int(args.batch_size)):
                batch_idx = perm[start : start + int(args.batch_size)]
                x = torch.as_tensor(inputs[batch_idx], dtype=torch.float32, device=device)
                intervene_t = torch.as_tensor(targets["intervene_target"][batch_idx], dtype=torch.float32, device=device)
                critical_t = torch.as_tensor(targets["critical_target"][batch_idx], dtype=torch.float32, device=device)
                risk_side_t = torch.as_tensor(arrays["risk_side"][batch_idx], dtype=torch.long, device=device)
                risk_reason_t = torch.as_tensor(arrays["risk_reason"][batch_idx], dtype=torch.long, device=device)
                intervene_weight_t = torch.as_tensor(targets["intervene_weight"][batch_idx], dtype=torch.float32, device=device)
                critical_weight_t = torch.as_tensor(targets["critical_weight"][batch_idx], dtype=torch.float32, device=device)
                side_weight_t = torch.as_tensor(targets["side_weight"][batch_idx], dtype=torch.float32, device=device)
                reason_weight_t = torch.as_tensor(targets["reason_weight"][batch_idx], dtype=torch.float32, device=device)

                pred = classifier(x)
                intervene_loss_per = F.binary_cross_entropy_with_logits(pred["intervene_logit"], intervene_t, reduction="none")
                side_loss_per = F.cross_entropy(pred["risk_side_logits"], risk_side_t, reduction="none")
                reason_loss_per = F.cross_entropy(pred["risk_reason_logits"], risk_reason_t, reduction="none")
                critical_loss_per = F.binary_cross_entropy_with_logits(pred["critical_logit"], critical_t, reduction="none")
                loss_intervene = weighted_mean(intervene_loss_per, intervene_weight_t)
                loss_side = weighted_mean(side_loss_per, side_weight_t)
                loss_reason = weighted_mean(reason_loss_per, reason_weight_t)
                loss_critical = weighted_mean(critical_loss_per, critical_weight_t)

                if args.mode == LEGACY_MODE:
                    loss = loss_intervene + 0.5 * loss_side + 0.5 * loss_reason + 0.5 * loss_critical
                elif args.mode == A3_MODE:
                    loss = 2.5 * loss_intervene + 4.0 * loss_critical + 0.05 * loss_side + 0.05 * loss_reason
                else:
                    loss = 3.0 * loss_intervene + 6.0 * loss_critical + 0.1 * loss_side + 0.1 * loss_reason

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.item())
                epoch_intervene += float(loss_intervene.item())
                epoch_side += float(loss_side.item())
                epoch_reason += float(loss_reason.item())
                epoch_critical += float(loss_critical.item())
                batch_count += 1

            append_csv_row(
                loss_csv,
                {
                    "epoch": epoch,
                    "loss": epoch_loss / max(batch_count, 1),
                    "intervene_loss": epoch_intervene / max(batch_count, 1),
                    "risk_side_loss": epoch_side / max(batch_count, 1),
                    "risk_reason_loss": epoch_reason / max(batch_count, 1),
                    "critical_loss": epoch_critical / max(batch_count, 1),
                },
            )

        offline = compute_offline_metrics(
            classifier,
            inputs,
            arrays,
            splits,
            threshold_intervene=float(args.threshold),
            threshold_critical=float(args.threshold),
        )
        include_full = (epoch == 0) or (epoch % int(args.eval_full_every) == 0) or (epoch == int(args.epochs))

        protocol_hard = evaluate_classifier_protocol(
            base_model=base_model,
            classifier=classifier,
            dataset=eval_dataset,
            cfg=cfg,
            traj_indices=hard_indices,
            shield_params=shield_params,
            tau_intervene=float(args.threshold),
            tau_critical=float(args.threshold),
            device=device,
            critical_overrides_gate=False,
            use_raw_critical_rule=False,
        )
        protocol_full = {
            "merge_success_rate": "",
            "endpoint_success_rate": "",
            "safety_success_rate": "",
            "collision_rate": "",
            "collision_count": "",
            "predicted_intervention_rate": "",
            "teacher_intervention_rate": "",
            "false_negative_rate": "",
            "critical_false_negative_rate": "",
        }
        if include_full:
            protocol_full = evaluate_classifier_protocol(
                base_model=base_model,
                classifier=classifier,
                dataset=eval_dataset,
                cfg=cfg,
                traj_indices=full_indices,
                shield_params=shield_params,
                tau_intervene=float(args.threshold),
                tau_critical=float(args.threshold),
                device=device,
                critical_overrides_gate=False,
                use_raw_critical_rule=False,
            )

        row = {
            "epoch": epoch,
            **offline,
            "oracle_full_false_negative_rate": oracle_fn_full,
            "oracle_hard_false_negative_rate": oracle_fn_hard,
            "hard15_protocol_merge_success_rate": protocol_hard["merge_success_rate"],
            "hard15_protocol_endpoint_success_rate": protocol_hard["endpoint_success_rate"],
            "hard15_protocol_safety_success_rate": protocol_hard["safety_success_rate"],
            "hard15_protocol_collision_rate": protocol_hard["collision_rate"],
            "hard15_protocol_false_negative_rate": protocol_hard["false_negative_rate"],
            "hard15_protocol_critical_false_negative_rate": protocol_hard["critical_false_negative_rate"],
            "hard15_protocol_predicted_intervention_rate": protocol_hard["predicted_intervention_rate"],
            "hard15_protocol_teacher_intervention_rate": protocol_hard["teacher_intervention_rate"],
            "full217_protocol_merge_success_rate": protocol_full["merge_success_rate"],
            "full217_protocol_endpoint_success_rate": protocol_full["endpoint_success_rate"],
            "full217_protocol_safety_success_rate": protocol_full["safety_success_rate"],
            "full217_protocol_collision_rate": protocol_full["collision_rate"],
            "full217_protocol_false_negative_rate": protocol_full["false_negative_rate"],
            "full217_protocol_critical_false_negative_rate": protocol_full["critical_false_negative_rate"],
            "full217_protocol_predicted_intervention_rate": protocol_full["predicted_intervention_rate"],
            "full217_protocol_teacher_intervention_rate": protocol_full["teacher_intervention_rate"],
        }
        append_csv_row(eval_csv, row)

        if epoch in save_epochs:
            ckpt_path = save_epoch_checkpoint(output_dir, classifier, epoch)
            saved_checkpoint_paths[str(epoch)] = str(ckpt_path.resolve())

        if include_full:
            current_key = best_key(offline, protocol_full, protocol_hard)
            if best is None or current_key > best:
                best = current_key
                best_epoch = epoch
                checkpoint_path = output_dir / "best_checkpoint.pt"
                with checkpoint_path.open("wb") as f:
                    torch.save({"state_dict": classifier.state_dict(), "epoch": epoch}, f)
                save_json_common(
                    output_dir / "best_model_info.json",
                    {
                        "best_epoch": epoch,
                        "best_key": list(current_key),
                        "offline_metrics": offline,
                        "protocol_hard": protocol_hard,
                        "protocol_full": protocol_full,
                        "oracle_full_false_negative_rate": oracle_fn_full,
                        "oracle_hard_false_negative_rate": oracle_fn_hard,
                        "checkpoint": str(checkpoint_path.resolve()),
                    },
                )
                print(f"[*] risk classifier best updated at epoch {epoch}")

    save_json_common(
        output_dir / "run_summary.json",
        {
            "mode": args.mode,
            "best_epoch": best_epoch,
            "best_checkpoint": str((output_dir / "best_checkpoint.pt").resolve()) if best_epoch is not None else "",
            "saved_checkpoints": saved_checkpoint_paths,
        },
    )


if __name__ == "__main__":
    main()
