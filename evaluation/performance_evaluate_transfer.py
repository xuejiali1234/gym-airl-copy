import argparse
import glob
import json
import os
import random
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from tqdm import tqdm

curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(curr_dir)
sys.path.append(root_dir)

from configs.config import Config
from envs.merging_env import MergingEnv
from utils.data_loader import MergingDataset


class SingleTrajDataset:
    def __init__(self, traj, expert_mean, expert_std):
        self.trajectories = [traj]
        self.expert_mean = expert_mean
        self.expert_std = expert_std

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.trajectories[idx]


def iter_csv_files(paths):
    all_files = []
    for path in paths:
        if os.path.isfile(path):
            all_files.append(path)
        elif os.path.isdir(path):
            all_files.extend(glob.glob(os.path.join(path, "**", "*.csv"), recursive=True))
    return sorted(list(set(all_files)))


def load_stats_dataset(stats_paths, device):
    dataset = MergingDataset(stats_paths, device=device)
    if len(dataset) == 0:
        raise ValueError(f"Stats dataset is empty: {stats_paths}")
    return dataset, dataset.get_stats()


def normalize_trajectories_with_fixed_stats(dataset, expert_mean, expert_std):
    dataset.expert_mean = expert_mean.astype(np.float32).copy()
    dataset.expert_std = expert_std.astype(np.float32).copy()

    for traj in dataset.trajectories:
        raw_states = traj["state"].copy()
        raw_next_states = traj["next_state"].copy()

        traj["state"] = (traj["state"] - dataset.expert_mean) / dataset.expert_std
        traj["next_state"] = (traj["next_state"] - dataset.expert_mean) / dataset.expert_std

        for idx in range(len(traj["state"])):
            if np.abs(raw_states[idx, 4:8]).sum() < 1e-5:
                traj["state"][idx, 4:8] = 0.0
            if np.abs(raw_next_states[idx, 4:8]).sum() < 1e-5:
                traj["next_state"][idx, 4:8] = 0.0

            if np.abs(raw_states[idx, 8:12]).sum() < 1e-5:
                traj["state"][idx, 8:12] = 0.0
            if np.abs(raw_next_states[idx, 8:12]).sum() < 1e-5:
                traj["next_state"][idx, 8:12] = 0.0

            if np.abs(raw_states[idx, 12:16]).sum() < 1e-5:
                traj["state"][idx, 12:16] = 0.0
            if np.abs(raw_next_states[idx, 12:16]).sum() < 1e-5:
                traj["next_state"][idx, 12:16] = 0.0

        goal_mean = dataset.expert_mean[0:2]
        goal_std = dataset.expert_std[0:2]
        traj["goal"] = (traj["goal"] - goal_mean) / goal_std


def load_eval_dataset_with_fixed_stats(eval_paths, expert_mean, expert_std, device):
    dataset = MergingDataset.__new__(MergingDataset)
    dataset.device = device
    dataset.trajectories = []
    dataset.cfg = Config()
    dataset.expert_mean = np.zeros(16, dtype=np.float32)
    dataset.expert_std = np.ones(16, dtype=np.float32)

    files = iter_csv_files(eval_paths)
    if not files:
        raise ValueError(f"No CSV files found in eval paths: {eval_paths}")

    for csv_file in files:
        df = pd.read_csv(csv_file)
        dataset._process_data(df, filename=os.path.basename(csv_file))

    if len(dataset.trajectories) == 0:
        raise ValueError(f"Processed 0 trajectories from eval paths: {eval_paths}")

    normalize_trajectories_with_fixed_stats(dataset, expert_mean, expert_std)
    dataset.confidence_weights = np.ones(len(dataset.trajectories), dtype=np.float32)
    return dataset


class TransferMetricsEvaluator:
    def __init__(self, model_path, stats_data_paths, eval_data_paths, collision_margin=1.0, seed=44):
        self.cfg = Config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ft_to_m = 0.3048
        self.collision_margin = collision_margin

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.stats_dataset, (self.expert_mean, self.expert_std) = load_stats_dataset(
            stats_data_paths,
            device=self.device,
        )
        self.eval_dataset = load_eval_dataset_with_fixed_stats(
            eval_data_paths,
            expert_mean=self.expert_mean,
            expert_std=self.expert_std,
            device=self.device,
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.model = PPO.load(model_path, device=self.device)
        self.model_path = model_path
        self.stats_data_paths = stats_data_paths
        self.eval_data_paths = eval_data_paths

    def evaluate(self, num_episodes=None):
        total_eps = len(self.eval_dataset)
        if num_episodes is None or num_episodes > total_eps:
            num_episodes = total_eps

        indices = list(range(total_eps))
        if num_episodes < total_eps:
            indices = random.sample(indices, num_episodes)

        stats = {"success": 0, "collision": 0, "timeout": 0, "total": 0}
        metrics_log = {"speeds": [], "jerks": [], "ttcs": [], "thws": []}

        for idx in tqdm(indices, desc="Transfer eval"):
            traj = self.eval_dataset[idx]
            single_dataset = SingleTrajDataset(traj, self.expert_mean, self.expert_std)
            env = MergingEnv(single_dataset)
            env.collision_margin = self.collision_margin

            obs, info = env.reset()
            done = False
            truncated = False
            step_count = 0
            max_steps = len(env.current_traj["ego_pos"]) + 50

            traj_speeds = []
            traj_jerks = []

            px, py, vx, vy = env.ego_state
            last_v = np.sqrt(vx ** 2 + vy ** 2)
            last_acc = 0.0

            min_ttc = float("inf")
            min_thw = float("inf")

            episode_collided = False
            is_endpoint_success = False

            while not (done or truncated) and step_count < max_steps:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                episode_collided = episode_collided or bool(info.get("is_collided", False))

                px, py, vx, vy = env.ego_state
                v_curr = np.sqrt(vx ** 2 + vy ** 2)
                acc_curr = (v_curr - last_v) / self.cfg.DT
                jerk_curr = abs(acc_curr - last_acc) / self.cfg.DT

                traj_speeds.append(v_curr)
                if step_count > 0:
                    traj_jerks.append(jerk_curr)

                last_v = v_curr
                last_acc = acc_curr

                surr = env._get_surround_at_t(env.t)
                ego_front_y = py + self.cfg.VEHICLE_LENGTH / 2.0

                def check_vehicle_ttc(veh_idx):
                    base_idx = veh_idx * 4
                    if surr[base_idx] == 0 and surr[base_idx + 1] == 0:
                        return float("inf"), float("inf")

                    target_rear_y = surr[base_idx + 1] - self.cfg.VEHICLE_LENGTH / 2.0
                    dist = target_rear_y - ego_front_y
                    if dist > 0 and abs(surr[base_idx] - px) < self.cfg.LANE_WIDTH:
                        rel_v = v_curr - surr[base_idx + 2]
                        thw = dist / v_curr if v_curr > 1.0 else float("inf")
                        ttc = dist / rel_v if rel_v > 0.1 else float("inf")
                        return ttc, thw
                    return float("inf"), float("inf")

                ttc1, thw1 = check_vehicle_ttc(0)
                ttc2, thw2 = check_vehicle_ttc(1)
                min_ttc = min(min_ttc, ttc1, ttc2)
                min_thw = min(min_thw, thw1, thw2)

                step_count += 1
                if done or truncated:
                    is_endpoint_success = info.get("is_endpoint_success", info.get("is_success", False))
                    episode_collided = episode_collided or bool(
                        getattr(env, "has_collided_this_episode", False)
                    )

            stats["total"] += 1
            if is_endpoint_success:
                stats["success"] += 1
            if episode_collided:
                stats["collision"] += 1
            if not is_endpoint_success:
                stats["timeout"] += 1

            if traj_speeds:
                metrics_log["speeds"].append(float(np.mean(traj_speeds)))
            if traj_jerks:
                metrics_log["jerks"].append(float(np.mean(traj_jerks)))
            if min_ttc != float("inf"):
                metrics_log["ttcs"].append(float(min_ttc))
            if min_thw != float("inf"):
                metrics_log["thws"].append(float(min_thw))

        return self.build_report(stats, metrics_log, num_episodes)

    def build_report(self, stats, metrics, num_episodes):
        total = stats["total"]
        success_rate = stats["success"] / total if total else 0.0
        collision_rate = stats["collision"] / total if total else 0.0
        timeout_rate = stats["timeout"] / total if total else 0.0

        report = {
            "model_path": self.model_path,
            "stats_data_paths": self.stats_data_paths,
            "eval_data_paths": self.eval_data_paths,
            "episodes_evaluated": total,
            "success_rate": success_rate,
            "collision_rate": collision_rate,
            "timeout_or_stuck_rate": timeout_rate,
            "avg_speed_mps": (float(np.mean(metrics["speeds"])) * self.ft_to_m) if metrics["speeds"] else 0.0,
            "avg_jerk_mps3": (float(np.mean(metrics["jerks"])) * self.ft_to_m) if metrics["jerks"] else 0.0,
            "avg_ttc_min_s": float(np.mean(metrics["ttcs"])) if metrics["ttcs"] else 0.0,
            "avg_thw_min_s": float(np.mean(metrics["thws"])) if metrics["thws"] else 0.0,
            "collision_margin_ft": self.collision_margin,
        }
        return report

    @staticmethod
    def print_report(report, title="TRANSFER EVALUATION REPORT"):
        print("\n" + "=" * 70)
        print(f"{title} (N={report['episodes_evaluated']})")
        print("=" * 70)
        print(f"{'Metric':<28} | {'Value':<15} | {'Unit':<10}")
        print("-" * 70)
        print(f"{'Success Rate':<28} | {report['success_rate']:.3f}           | -")
        print(f"{'Collision Rate':<28} | {report['collision_rate']:.3f}           | -")
        print(f"{'Timeout/Stuck Rate':<28} | {report['timeout_or_stuck_rate']:.3f}           | -")
        print("-" * 70)
        print(f"{'Avg Speed':<28} | {report['avg_speed_mps']:.3f}           | m/s")
        print(f"{'Avg Jerk':<28} | {report['avg_jerk_mps3']:.3f}           | m/s^3")
        print("-" * 70)
        print(f"{'Avg TTC (Min)':<28} | {report['avg_ttc_min_s']:.3f}           | s")
        print(f"{'Avg THW (Min)':<28} | {report['avg_thw_min_s']:.3f}           | s")
        print("=" * 70)


def build_default_stats_paths():
    return [
        os.path.join(root_dir, "data", "lane_change_trajectories-0750am-0805am"),
        os.path.join(root_dir, "data", "lane_change_trajectories-0805am-0820am"),
        os.path.join(root_dir, "data", "lane_change_trajectories-0820am-0835am"),
    ]


def build_default_model_path():
    return os.path.join(
        root_dir,
        "train_log",
        "baseline_attn_goal_safe_branch_aux_probe_P300_U220_L5e6_D230_20260425_224122",
        "checkpoints",
        "baseline_policy_attn_goal_safe_branch_aux_probe_P300_U220_L5e6_D230_epoch_290.zip",
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate transfer performance on a new dataset.")
    parser.add_argument("--model-path", default=build_default_model_path())
    parser.add_argument("--stats-data-path", nargs="+", default=build_default_stats_paths())
    parser.add_argument("--eval-data-path", nargs="+", required=True)
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument("--collision-margin", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--save-json", default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    evaluator = TransferMetricsEvaluator(
        model_path=args.model_path,
        stats_data_paths=args.stats_data_path,
        eval_data_paths=args.eval_data_path,
        collision_margin=args.collision_margin,
        seed=args.seed,
    )
    report = evaluator.evaluate(num_episodes=args.num_episodes)
    evaluator.print_report(report, title="I80 TRANSFER EVALUATION REPORT")

    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json), exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Saved report: {args.save_json}")


if __name__ == "__main__":
    main()
