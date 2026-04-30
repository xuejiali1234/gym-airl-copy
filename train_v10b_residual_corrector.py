import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Sequence, Tuple

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
from evaluation.safety_shield_evaluate import shield_action, target_lane_vehicle_metrics
from model.attention_net import AttentionFeaturesExtractor, GoalConditionedMLPFeaturesExtractor  # noqa: F401
from v10b_ab_common import (
    append_csv_row,
    build_ab_input_matrix,
    make_classification_arrays,
    sanitize_numeric,
    save_json,
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


class ResidualCorrector(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.delta_head = nn.Linear(128, 2)
        self.gate_head = nn.Linear(128, 1)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, -4.0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.encoder(x)
        return {
            "delta_action": self.delta_head(feat),
            "gate_logit": self.gate_head(feat).squeeze(-1),
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Train a gated residual corrector on top of the frozen base policy.")
    parser.add_argument("--teacher-dataset", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval-full-every", type=int, default=5)
    parser.add_argument("--eval-full-limit", type=int, default=0)
    parser.add_argument("--eval-hard-limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def default_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT_DIR / "train_log" / f"v10b_residual_corrector_{timestamp}"


def evaluate_base_policy_suite(base_model: PPO, dataset, cfg: Config, traj_indices: Sequence[int], shield_params: Dict) -> Tuple[Dict[str, float], Dict[str, float]]:
    from evaluation.failure_case_full_evaluate import evaluate_single_trajectory, summarize_rows
    from evaluation.safety_shield_evaluate import evaluate_single_trajectory_with_shield, summarize_with_shield

    model_info = get_model_info_by_tag(BASE_MODEL_TAG)
    no_rows = [{**evaluate_single_trajectory(base_model, dataset, cfg, traj_index), "model_tag": BASE_MODEL_TAG} for traj_index in traj_indices]
    sh_rows = [{**evaluate_single_trajectory_with_shield(base_model, dataset, cfg, traj_index, shield_params), "model_tag": BASE_MODEL_TAG} for traj_index in traj_indices]
    no_summary = summarize_rows(model_info, no_rows)
    sh_summary = summarize_with_shield(model_info, sh_rows, "eval", {"endpoint": 0.95, "safety": 0.90, "collision": 0.01})
    return no_summary, sh_summary


def evaluate_residual_protocol(
    base_model: PPO,
    residual_model: ResidualCorrector,
    dataset,
    cfg: Config,
    traj_indices: Sequence[int],
    shield_params: Dict,
    device: torch.device,
    use_v10b: bool,
) -> Dict[str, float]:
    total = 0
    merge_success = 0
    endpoint_success = 0
    safety_success = 0
    collision_count = 0
    shield_interventions = 0
    total_steps = 0
    gate_mean_sum = 0.0
    gate_active_count = 0

    residual_model.eval()
    for traj_index in tqdm(traj_indices, desc="residual_eval", leave=False):
        traj = dataset[traj_index]
        env = MergingEnv(SingleTrajDataset(traj, dataset.expert_mean, dataset.expert_std))
        env.collision_margin = 1.0
        obs, _ = env.reset(seed=cfg.SEED + int(traj_index))
        terminated = False
        truncated = False
        max_steps = len(env.current_traj["ego_pos"]) + 50
        step_count = 0
        shield_count_ep = 0
        consecutive_interventions = 0
        safe_steps = 0
        last_info = {}

        while not (terminated or truncated) and step_count < max_steps:
            policy_action, _ = base_model.predict(obs, deterministic=True)
            policy_action = np.asarray(policy_action, dtype=np.float32).reshape(-1)
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
            x = np.concatenate([sanitize_numeric(np.asarray(obs, dtype=np.float32)), sanitize_numeric(policy_action), sanitize_numeric(aux)], axis=0)[None, :]
            with torch.no_grad():
                pred = residual_model(torch.as_tensor(x, dtype=torch.float32, device=device))
                gate = torch.sigmoid(pred["gate_logit"]).item()
                delta = pred["delta_action"].squeeze(0).cpu().numpy()
            corrected_action = np.clip(policy_action + gate * delta, -1.0, 1.0).astype(np.float32)
            gate_mean_sum += gate
            gate_active_count += int(gate >= 0.5)

            if use_v10b:
                action, shield_info = shield_action(
                    env=env,
                    policy_action=corrected_action,
                    params=shield_params,
                    step_count=step_count,
                    shield_interventions=shield_count_ep,
                    consecutive_interventions=consecutive_interventions,
                    safe_steps=safe_steps,
                )
                action = np.asarray(action, dtype=np.float32).reshape(-1)
                if bool(shield_info["intervened"]):
                    shield_interventions += 1
                    shield_count_ep += 1
                    consecutive_interventions += 1
                else:
                    consecutive_interventions = 0
            else:
                action = corrected_action

            obs, _, terminated, truncated, info = env.step(action)
            last_info = dict(info)
            total_steps += 1

            if (
                use_v10b
                and not bool(getattr(env, "has_collided_this_episode", False))
                and hasattr(env, "_get_surround_at_t")
            ):
                surr_now = env._get_surround_at_t(env.t)
                px, py, _, vy = [float(x) for x in env.ego_state]
                min_ttc, min_thw = env._compute_min_ttc_thw(px, py, vy, surr_now)
                if (
                    min_ttc >= shield_params.get("lead_ttc_min", 3.0)
                    and min_thw >= shield_params.get("follow_thw_min", 0.8)
                ):
                    safe_steps += 1
                else:
                    safe_steps = 0
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
        "shield_intervention_rate": shield_interventions / max(total_steps, 1) if use_v10b else 0.0,
        "residual_gate_mean": gate_mean_sum / max(total_steps, 1),
        "residual_gate_active_rate": gate_active_count / max(total_steps, 1),
    }


def make_weights(arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    hard = np.where(arrays["hard15"], 5.0, 1.0).astype(np.float32)
    intervene = arrays["intervene"] > 0.5
    warning = arrays["warning"]
    normal_safe = arrays["normal_safe"]
    return {
        "delta_weight": hard * intervene.astype(np.float32),
        "zero_weight": hard * np.where(warning, 0.5, np.where(normal_safe, 1.0, 0.0)).astype(np.float32),
        "gate_weight": hard * np.where(intervene, 1.0, np.where(warning, 0.5, 1.0)).astype(np.float32),
    }


def best_key(residual_no_shield_full: Dict[str, float], residual_shield_full: Dict[str, float]) -> Tuple[float, float, float, float, float]:
    zero_collision_pass = 1.0 if float(residual_shield_full["collision_rate"]) <= 1e-12 else 0.0
    return (
        zero_collision_pass,
        -float(residual_shield_full["shield_intervention_rate"]),
        -float(residual_no_shield_full["collision_rate"]),
        float(residual_shield_full["endpoint_success_rate"]),
        float(residual_shield_full["merge_success_rate"]),
    )


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else default_output_dir().resolve()
    teacher_payload = load_teacher_payload(Path(args.teacher_dataset).resolve())
    samples = teacher_payload["samples"]
    seed_everything(args.seed, deterministic=True)

    if args.dry_run:
        print("=" * 80)
        print("v10b residual corrector dry-run")
        print(f"teacher_dataset={Path(args.teacher_dataset).resolve()}")
        print(f"output_dir={output_dir}")
        print(f"samples={len(samples)}")
        print("=" * 80)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()
    shield_params = build_v10b_shield_params()
    eval_dataset = load_eval_dataset(device=device)
    base_model = PPO.load(str(get_model_info_by_tag(BASE_MODEL_TAG)["checkpoint"]), device=device)

    inputs = build_ab_input_matrix(samples)
    arrays = make_classification_arrays(samples)
    weights = make_weights(arrays)
    splits = teacher_split_indices(samples, seed=args.seed)
    residual_model = ResidualCorrector(input_dim=inputs.shape[1]).to(device)
    optimizer = Adam(residual_model.parameters(), lr=float(args.lr))

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
    save_json_common(
        output_dir / "config.json",
        {
            "teacher_dataset": str(Path(args.teacher_dataset).resolve()),
            "seed": int(args.seed),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "train_samples": int(len(splits["train"])),
            "val_samples": int(len(splits["val"])),
            "hard_samples": int(len(splits["hard"])),
        },
    )

    hard_traj_indices = sorted(set(int(samples[idx]["traj_index"]) for idx in splits["hard"]))
    full_traj_indices = list(range(len(eval_dataset)))
    if args.eval_hard_limit:
        hard_traj_indices = hard_traj_indices[: int(args.eval_hard_limit)]
    if args.eval_full_limit:
        full_traj_indices = full_traj_indices[: int(args.eval_full_limit)]

    base_no_hard, base_shield_hard = evaluate_base_policy_suite(base_model, eval_dataset, cfg, hard_traj_indices, shield_params)
    base_no_full, base_shield_full = evaluate_base_policy_suite(base_model, eval_dataset, cfg, full_traj_indices, shield_params)

    loss_csv = output_dir / "loss.csv"
    eval_csv = output_dir / "eval_summary.csv"
    best = None
    best_epoch = None

    for epoch in range(0, int(args.epochs) + 1):
        if epoch > 0:
            residual_model.train()
            perm = np.random.permutation(splits["train"])
            total_loss = 0.0
            delta_loss_sum = 0.0
            zero_loss_sum = 0.0
            gate_loss_sum = 0.0
            mag_loss_sum = 0.0
            batch_count = 0

            for start in range(0, len(perm), int(args.batch_size)):
                batch_idx = perm[start : start + int(args.batch_size)]
                x = torch.as_tensor(inputs[batch_idx], dtype=torch.float32, device=device)
                delta_target = torch.as_tensor(arrays["delta_action"][batch_idx], dtype=torch.float32, device=device)
                gate_target = torch.as_tensor(arrays["intervene"][batch_idx], dtype=torch.float32, device=device)
                delta_weight = torch.as_tensor(weights["delta_weight"][batch_idx], dtype=torch.float32, device=device)
                zero_weight = torch.as_tensor(weights["zero_weight"][batch_idx], dtype=torch.float32, device=device)
                gate_weight = torch.as_tensor(weights["gate_weight"][batch_idx], dtype=torch.float32, device=device)

                pred = residual_model(x)
                delta_pred = pred["delta_action"]
                gate_logit = pred["gate_logit"]
                delta_err = ((delta_pred - delta_target) ** 2).mean(dim=1)
                zero_err = (delta_pred ** 2).mean(dim=1)
                gate_err = F.binary_cross_entropy_with_logits(gate_logit, gate_target, reduction="none")
                mag_loss = (delta_pred ** 2).mean(dim=1)

                loss_delta = (delta_weight * delta_err).sum() / delta_weight.sum().clamp_min(1e-6)
                loss_zero = (zero_weight * zero_err).sum() / zero_weight.sum().clamp_min(1e-6)
                loss_gate = (gate_weight * gate_err).sum() / gate_weight.sum().clamp_min(1e-6)
                loss_mag = mag_loss.mean()
                loss = loss_delta + loss_zero + loss_gate + 0.01 * loss_mag

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item())
                delta_loss_sum += float(loss_delta.item())
                zero_loss_sum += float(loss_zero.item())
                gate_loss_sum += float(loss_gate.item())
                mag_loss_sum += float(loss_mag.item())
                batch_count += 1

            append_csv_row(
                loss_csv,
                {
                    "epoch": epoch,
                    "loss": total_loss / max(batch_count, 1),
                    "loss_delta": delta_loss_sum / max(batch_count, 1),
                    "loss_zero": zero_loss_sum / max(batch_count, 1),
                    "loss_gate": gate_loss_sum / max(batch_count, 1),
                    "loss_mag": mag_loss_sum / max(batch_count, 1),
                },
            )

        include_full = (epoch == 0) or (epoch % int(args.eval_full_every) == 0) or (epoch == int(args.epochs))
        residual_no_hard = evaluate_residual_protocol(
            base_model, residual_model, eval_dataset, cfg, hard_traj_indices, shield_params, device, use_v10b=False
        )
        residual_shield_hard = evaluate_residual_protocol(
            base_model, residual_model, eval_dataset, cfg, hard_traj_indices, shield_params, device, use_v10b=True
        )
        residual_no_full = {}
        residual_shield_full = {}
        if include_full:
            residual_no_full = evaluate_residual_protocol(
                base_model, residual_model, eval_dataset, cfg, full_traj_indices, shield_params, device, use_v10b=False
            )
            residual_shield_full = evaluate_residual_protocol(
                base_model, residual_model, eval_dataset, cfg, full_traj_indices, shield_params, device, use_v10b=True
            )

        row = {
            "epoch": epoch,
            "base_hard_no_shield_collision_rate": base_no_hard["collision_rate"],
            "base_hard_no_shield_merge_success_rate": base_no_hard["merge_success_rate"],
            "base_hard_no_shield_endpoint_success_rate": base_no_hard["endpoint_success_rate"],
            "base_hard_no_shield_safety_success_rate": base_no_hard["safety_success_rate"],
            "base_hard_shield_collision_rate": base_shield_hard["collision_rate"],
            "base_hard_shield_merge_success_rate": base_shield_hard["merge_success_rate"],
            "base_hard_shield_endpoint_success_rate": base_shield_hard["endpoint_success_rate"],
            "base_hard_shield_safety_success_rate": base_shield_hard["safety_success_rate"],
            "residual_hard_no_shield_collision_rate": residual_no_hard["collision_rate"],
            "residual_hard_no_shield_merge_success_rate": residual_no_hard["merge_success_rate"],
            "residual_hard_no_shield_endpoint_success_rate": residual_no_hard["endpoint_success_rate"],
            "residual_hard_no_shield_safety_success_rate": residual_no_hard["safety_success_rate"],
            "residual_hard_shield_collision_rate": residual_shield_hard["collision_rate"],
            "residual_hard_shield_merge_success_rate": residual_shield_hard["merge_success_rate"],
            "residual_hard_shield_endpoint_success_rate": residual_shield_hard["endpoint_success_rate"],
            "residual_hard_shield_safety_success_rate": residual_shield_hard["safety_success_rate"],
            "residual_hard_shield_rate": residual_shield_hard["shield_intervention_rate"],
            "residual_hard_gate_mean": residual_shield_hard["residual_gate_mean"],
            "residual_hard_gate_active_rate": residual_shield_hard["residual_gate_active_rate"],
            "base_full_no_shield_collision_rate": base_no_full["collision_rate"] if include_full else "",
            "base_full_no_shield_merge_success_rate": base_no_full["merge_success_rate"] if include_full else "",
            "base_full_no_shield_endpoint_success_rate": base_no_full["endpoint_success_rate"] if include_full else "",
            "base_full_no_shield_safety_success_rate": base_no_full["safety_success_rate"] if include_full else "",
            "base_full_shield_collision_rate": base_shield_full["collision_rate"] if include_full else "",
            "base_full_shield_merge_success_rate": base_shield_full["merge_success_rate"] if include_full else "",
            "base_full_shield_endpoint_success_rate": base_shield_full["endpoint_success_rate"] if include_full else "",
            "base_full_shield_safety_success_rate": base_shield_full["safety_success_rate"] if include_full else "",
            "base_full_shield_shield_rate": base_shield_full.get("shield_intervention_rate", "") if include_full else "",
            "residual_full_no_shield_collision_rate": residual_no_full.get("collision_rate", ""),
            "residual_full_no_shield_merge_success_rate": residual_no_full.get("merge_success_rate", ""),
            "residual_full_no_shield_endpoint_success_rate": residual_no_full.get("endpoint_success_rate", ""),
            "residual_full_no_shield_safety_success_rate": residual_no_full.get("safety_success_rate", ""),
            "residual_full_shield_collision_rate": residual_shield_full.get("collision_rate", ""),
            "residual_full_shield_merge_success_rate": residual_shield_full.get("merge_success_rate", ""),
            "residual_full_shield_endpoint_success_rate": residual_shield_full.get("endpoint_success_rate", ""),
            "residual_full_shield_safety_success_rate": residual_shield_full.get("safety_success_rate", ""),
            "residual_full_shield_rate": residual_shield_full.get("shield_intervention_rate", ""),
            "residual_full_gate_mean": residual_shield_full.get("residual_gate_mean", ""),
            "residual_full_gate_active_rate": residual_shield_full.get("residual_gate_active_rate", ""),
        }
        append_csv_row(eval_csv, row)

        if include_full:
            current_key = best_key(residual_no_full, residual_shield_full)
            if best is None or current_key > best:
                best = current_key
                best_epoch = epoch
                checkpoint_path = output_dir / "best_checkpoint.pt"
                with checkpoint_path.open("wb") as f:
                    torch.save({"state_dict": residual_model.state_dict(), "epoch": epoch}, f)
                save_json_common(
                    output_dir / "best_model_info.json",
                    {
                        "best_epoch": epoch,
                        "best_key": list(current_key),
                        "base_no_full": base_no_full,
                        "base_shield_full": base_shield_full,
                        "residual_no_full": residual_no_full,
                        "residual_shield_full": residual_shield_full,
                        "base_no_hard": base_no_hard,
                        "base_shield_hard": base_shield_hard,
                        "residual_no_hard": residual_no_hard,
                        "residual_shield_hard": residual_shield_hard,
                        "checkpoint": str(checkpoint_path.resolve()),
                    },
                )
                print(f"[*] residual corrector best updated at epoch {epoch}")

    save_json_common(
        output_dir / "run_summary.json",
        {
            "best_epoch": best_epoch,
            "best_checkpoint": str((output_dir / "best_checkpoint.pt").resolve()) if best_epoch is not None else "",
        },
    )


if __name__ == "__main__":
    main()
