import argparse
import os
import random
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from stable_baselines3 import PPO
from tqdm import tqdm


curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(curr_dir)
sys.path.append(root_dir)

from configs.config import Config
from envs.merging_env import MergingEnv
from model.attention_net import AttentionFeaturesExtractor, GoalConditionedMLPFeaturesExtractor
from utils.data_loader import MergingDataset


try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        plt.style.use("ggplot")

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False

TITLE_FONTSIZE = 28
LABEL_FONTSIZE = 24
TICK_FONTSIZE = 20
LEGEND_FONTSIZE = 20

DEFAULT_MODEL_TAG = "PPO_RL_Goal_PushMerge_E300_epoch290"
DEFAULT_MODEL_PATH = os.path.join(
    root_dir,
    "train_log",
    "ppo_rl_baseline_PPO_RL_Goal_PushMerge_E300_20260622_122052",
    "checkpoints",
    "ppo_rl_policy_PPO_RL_Goal_PushMerge_E300_epoch_290.zip",
)
DEFAULT_OUTPUT_DIR = os.path.join(curr_dir, "distribution_comparison_results_rl")


class SingleTrajDataset:
    def __init__(self, traj, expert_mean, expert_std):
        self.trajectories = [traj]
        self.expert_mean = expert_mean
        self.expert_std = expert_std

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.trajectories[idx]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RL policy trajectory distributions against real data.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="RL PPO checkpoint path.")
    parser.add_argument("--model-tag", default=DEFAULT_MODEL_TAG, help="Label shown in legends and output names.")
    parser.add_argument("--n-samples", type=int, default=217, help="Number of trajectories to evaluate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for trajectory sampling.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for png/csv outputs.")
    parser.add_argument("--show", action="store_true", help="Display the figure window after saving.")
    return parser.parse_args()


def build_rl_cfg():
    cfg = Config()
    cfg.ENABLE_GOAL_CONDITION = True
    cfg.GOAL_ABLATION_MODE = "normal"
    cfg.ENABLE_ATTENTION = False
    cfg.ATTENTION_ABLATION_MODE = "normal"
    cfg.ENABLE_SAFETY_MODULE = False
    cfg.ENABLE_SAFETY_BRANCH = False
    cfg.ENABLE_SAFETY_AUX_LOSS = False
    cfg.ENABLE_PREDICTIVE_SAFETY_CRITIC = False
    cfg.ENABLE_PREDICTIVE_SAFETY_RESIDUAL = False
    cfg.PREDICTIVE_SAFETY_ENABLE_CPAIR_ADDITIVE = False
    return cfg


class Figure5GeneratorRL:
    def __init__(self, model_path: str, model_tag: str, seed: int):
        self.cfg = build_rl_cfg()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ft_to_m = 0.3048
        self.model_path = model_path
        self.model_tag = model_tag
        self.seed = seed

        self.data_paths = [
            os.path.join(root_dir, "data", "lane_change_trajectories-0750am-0805am"),
            os.path.join(root_dir, "data", "lane_change_trajectories-0805am-0820am"),
            os.path.join(root_dir, "data", "lane_change_trajectories-0820am-0835am"),
        ]

        print("[*] Loading dataset for global normalization stats...")
        self.global_dataset = MergingDataset(self.data_paths, device=self.device)
        if len(self.global_dataset) == 0:
            raise ValueError("Dataset is empty.")
        self.expert_mean, self.expert_std = self.global_dataset.get_stats()
        print(f"[OK] Loaded dataset with {len(self.global_dataset)} trajectories.")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")
        print(f"[*] Loading RL model: {self.model_path}")
        _ = AttentionFeaturesExtractor
        _ = GoalConditionedMLPFeaturesExtractor
        self.model = PPO.load(self.model_path, device=self.device)
        print("[OK] RL model loaded.")

        self.target_lane_divider = self.cfg.X_MIN + getattr(self.cfg, "LANE_WIDTH", 12.0)
        self.car_len = getattr(self.cfg, "VEHICLE_LENGTH", 15.0)

    def _calculate_traj_score(self, df: pd.DataFrame) -> float:
        vy = df["KF_Vel_Y"]
        efficiency = vy.mean() - vy.std()

        if "KF_Acc_Y" in df:
            acc_y = df["KF_Acc_Y"].values
        else:
            acc_y = np.diff(df["KF_Vel_Y"].values, prepend=df["KF_Vel_Y"].values[0]) / self.cfg.DT

        jerk = np.diff(acc_y) / self.cfg.DT
        comfort = np.abs(jerk).mean()

        if "L6_Leading_Local_Y" in df:
            dist = df["L6_Leading_Local_Y"] - df["KF_Local_Y"]
            rel_vel = df["KF_Vel_Y"] - df["L6_Leading_Vel"]
            mask = (rel_vel > 0.1) & (dist > 0)
            if mask.sum() > 0:
                ttc = dist[mask] / rel_vel[mask]
                ttc = np.clip(ttc, 0, 20.0)
                safety = ttc.mean()
            else:
                safety = 20.0
        else:
            safety = 20.0

        raw_score = 1.0 * efficiency + 1.0 * safety - 5.0 * comfort
        return raw_score / 100.0

    def calculate_metrics_frame(self, df_or_dict, is_dict=False) -> Dict[str, List[float]]:
        if not is_dict:
            py = df_or_dict["KF_Local_Y"].values
            vx = df_or_dict["KF_Vel_X"].values
            vy = df_or_dict["KF_Vel_Y"].values
            px = df_or_dict["KF_Local_X"].values
            l6_lead_y = df_or_dict["L6_Leading_Local_Y"].values
            l6_lead_v = df_or_dict["L6_Leading_Vel"].values
            l5_lead_y = df_or_dict["L5_Leading_Local_Y"].values
            l5_lead_v = df_or_dict["L5_Leading_Vel"].values
        else:
            py = np.array(df_or_dict["py"])
            px = np.array(df_or_dict["px"])
            vx = np.array(df_or_dict["vx"])
            vy = np.array(df_or_dict["vy"])
            l6_lead_y = np.array(df_or_dict["l6_lead_y"])
            l6_lead_v = np.array(df_or_dict["l6_lead_v"])
            l5_lead_y = np.array(df_or_dict["l5_lead_y"])
            l5_lead_v = np.array(df_or_dict["l5_lead_v"])

        ax = np.diff(vx, prepend=vx[0]) / self.cfg.DT
        ay = np.diff(vy, prepend=vy[0]) / self.cfg.DT
        jerk_x = np.diff(ax, prepend=ax[0]) / self.cfg.DT
        jerk_y = np.diff(ay, prepend=ay[0]) / self.cfg.DT

        speed = np.sqrt(vx ** 2 + vy ** 2) * self.ft_to_m
        acc_long = ay * self.ft_to_m
        lat_speed_abs = np.abs(vx) * self.ft_to_m
        jerk_2d = np.sqrt(jerk_x ** 2 + jerk_y ** 2) * self.ft_to_m

        ttc_list = []
        thw_list = []
        for i in range(len(py)):
            lead_y, lead_v = (l6_lead_y[i], l6_lead_v[i]) if px[i] > self.target_lane_divider else (l5_lead_y[i], l5_lead_v[i])
            if lead_y == 0:
                continue

            dist = lead_y - py[i] - self.car_len
            rel_v = vy[i] - lead_v

            if rel_v > 0.1 and dist > 0:
                val = dist / rel_v
                if val < 20:
                    ttc_list.append(val)

            if vy[i] > 1.0 and dist > 0:
                val = dist / vy[i]
                if val < 10:
                    thw_list.append(val)

        return {
            "speed": speed.tolist(),
            "acc_long": acc_long.tolist(),
            "lat_speed_abs": lat_speed_abs.tolist(),
            "jerk_2d": jerk_2d.tolist(),
            "ttc": ttc_list,
            "thw": thw_list,
        }

    def _build_real_df(self, traj) -> pd.DataFrame:
        ego_pos = traj["ego_pos"]
        ego_vel = traj["ego_vel"]
        surround_data = traj["surround_data"]
        return pd.DataFrame(
            {
                "KF_Local_X": ego_pos[:, 0],
                "KF_Local_Y": ego_pos[:, 1],
                "KF_Vel_X": ego_vel[:, 0],
                "KF_Vel_Y": ego_vel[:, 1],
                "L6_Leading_Local_X": surround_data[:, 0],
                "L6_Leading_Local_Y": surround_data[:, 1],
                "L6_Leading_Vel": surround_data[:, 2],
                "L5_Leading_Local_X": surround_data[:, 4],
                "L5_Leading_Local_Y": surround_data[:, 5],
                "L5_Leading_Vel": surround_data[:, 6],
            }
        )

    def _rollout_model(self, traj) -> Dict[str, List[float]]:
        single_dataset = SingleTrajDataset(traj, self.expert_mean, self.expert_std)
        env = MergingEnv(single_dataset, cfg=self.cfg)

        sim_res = {
            "px": [],
            "py": [],
            "vx": [],
            "vy": [],
            "l6_lead_y": [],
            "l6_lead_v": [],
            "l5_lead_y": [],
            "l5_lead_v": [],
        }

        obs, _ = env.reset()
        done = False
        truncated = False
        max_steps = len(traj["ego_pos"]) + 50
        step_count = 0

        while not (done or truncated) and step_count < max_steps:
            _, _, _, prev_vy = env.ego_state
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, done, truncated, _ = env.step(action)

            px, py, vx, vy = env.ego_state
            surr_now = env._get_surround_at_t(env.t)

            sim_res["px"].append(px)
            sim_res["py"].append(py)
            sim_res["vx"].append(vx)
            sim_res["vy"].append(vy)
            sim_res["l6_lead_y"].append(surr_now[1])
            sim_res["l6_lead_v"].append(surr_now[2])
            sim_res["l5_lead_y"].append(surr_now[5])
            sim_res["l5_lead_v"].append(surr_now[6])
            step_count += 1

        return sim_res

    def collect_data(self, n_samples=217):
        print(f"[*] Collecting distribution data with n_samples={n_samples} ...")
        all_traj_infos = []

        n_samples = min(n_samples, len(self.global_dataset))
        rng = random.Random(self.seed)
        selected_indices = rng.sample(range(len(self.global_dataset)), n_samples)

        print("[1/2] Processing real trajectories and ranking them...")
        for idx in tqdm(selected_indices):
            traj = self.global_dataset[idx]
            df = self._build_real_df(traj)
            score = self._calculate_traj_score(df)
            all_traj_infos.append(
                {
                    "traj": traj,
                    "df": df,
                    "score": score,
                    "metrics": self.calculate_metrics_frame(df, is_dict=False),
                }
            )

        all_traj_infos.sort(key=lambda x: x["score"], reverse=True)
        n_top20 = max(1, int(len(all_traj_infos) * 0.2))
        top20_infos = all_traj_infos[:n_top20]
        print(f"[OK] Selected Top 20% trajectories: {len(top20_infos)} / {len(all_traj_infos)}")

        metric_names = ("speed", "acc_long", "lat_speed_abs", "jerk_2d", "ttc", "thw")
        real_data_all = {name: [] for name in metric_names}
        real_data_top20 = {name: [] for name in metric_names}
        model_data = {name: [] for name in metric_names}

        for info in all_traj_infos:
            m = info["metrics"]
            for name in metric_names:
                real_data_all[name].extend(m[name])

        for info in top20_infos:
            m = info["metrics"]
            for name in metric_names:
                real_data_top20[name].extend(m[name])

        print("[2/2] Rolling out RL model...")
        for info in tqdm(all_traj_infos):
            sim_res = self._rollout_model(info["traj"])
            model_metrics = self.calculate_metrics_frame(sim_res, is_dict=True)
            for name in metric_names:
                model_data[name].extend(model_metrics[name])

        return real_data_all, real_data_top20, model_data

    def plot(self, real_data_all, real_data_top20, model_data, output_dir: str, show: bool):
        os.makedirs(output_dir, exist_ok=True)
        print("[*] Plotting distribution figure...")

        fig, axes = plt.subplots(2, 2, figsize=(24, 18))
        color_real = "gray"
        color_top20 = "green"
        color_model = "royalblue"
        alpha = 0.5
        bins = 30
        model_label = "PPO-RL"

        ax = axes[0, 0]
        sns.histplot(real_data_all["speed"], color=color_real, label="Real Data (All)", kde=True, ax=ax, stat="density", bins=bins, alpha=alpha)
        # sns.histplot(real_data_top20["speed"], color=color_top20, label="Real Data (Top 20%)", kde=True, ax=ax, stat="density", bins=bins, alpha=0.3)
        sns.histplot(model_data["speed"], color=color_model, label=model_label, kde=True, ax=ax, stat="density", bins=bins, alpha=alpha)
        ax.set_title("(a) Speed Distribution (m/s)", fontsize=TITLE_FONTSIZE, pad=14)
        ax.set_xlabel("Speed (m/s)", fontsize=LABEL_FONTSIZE)
        ax.set_ylabel("Density", fontsize=LABEL_FONTSIZE)
        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
        ax.legend(fontsize=LEGEND_FONTSIZE, loc="upper right", frameon=True)

        ax = axes[0, 1]
        sns.histplot(real_data_all["acc_long"], color=color_real, label="Real Data (All)", kde=True, ax=ax, stat="density", bins=bins, alpha=alpha)
        # sns.histplot(real_data_top20["acc_long"], color=color_top20, label="Real Data (Top 20%)", kde=True, ax=ax, stat="density", bins=bins, alpha=0.3)
        sns.histplot(model_data["acc_long"], color=color_model, label=model_label, kde=True, ax=ax, stat="density", bins=bins, alpha=alpha)
        ax.set_xlim(-4, 4)
        ax.set_title("(b) Acceleration Distribution (m/s^2)", fontsize=TITLE_FONTSIZE, pad=14)
        ax.set_xlabel("Acceleration (m/s^2)", fontsize=LABEL_FONTSIZE)
        ax.set_ylabel("Density", fontsize=LABEL_FONTSIZE)
        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
        ax.legend(fontsize=LEGEND_FONTSIZE, loc="upper right", frameon=True)

        ax = axes[1, 0]
        sns.kdeplot(real_data_all["ttc"], color=color_real, label="Real Data (All)", fill=True, ax=ax, alpha=0.2)
        # sns.kdeplot(real_data_top20["ttc"], color=color_top20, label="Real Data (Top 20%)", fill=True, ax=ax, alpha=0.2)
        sns.kdeplot(model_data["ttc"], color=color_model, label=model_label, fill=True, ax=ax, alpha=0.2)
        ax.set_title("(c) TTC Distribution (s)", fontsize=TITLE_FONTSIZE, pad=14)
        ax.set_xlabel("Time to Collision (s)", fontsize=LABEL_FONTSIZE)
        ax.set_ylabel("Density", fontsize=LABEL_FONTSIZE)
        ax.set_xlim(0, 15)
        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
        ax.legend(fontsize=LEGEND_FONTSIZE, loc="upper right", frameon=True)

        ax = axes[1, 1]
        sns.kdeplot(real_data_all["thw"], color=color_real, label="Real Data (All)", fill=True, ax=ax, alpha=0.2)
        # sns.kdeplot(real_data_top20["thw"], color=color_top20, label="Real Data (Top 20%)", fill=True, ax=ax, alpha=0.2)
        sns.kdeplot(model_data["thw"], color=color_model, label=model_label, fill=True, ax=ax, alpha=0.2)
        ax.set_title("(d) Time Headway Distribution (s)", fontsize=TITLE_FONTSIZE, pad=14)
        ax.set_xlabel("Time Headway (s)", fontsize=LABEL_FONTSIZE)
        ax.set_ylabel("Density", fontsize=LABEL_FONTSIZE)
        ax.set_xlim(0, 5)
        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
        ax.legend(fontsize=LEGEND_FONTSIZE, loc="upper right", frameon=True)

        plt.tight_layout(pad=3.6, w_pad=2.8, h_pad=3.0)
        fig_path = os.path.join(output_dir, f"{self.model_tag}_Distribution_Comparison.png")
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"[OK] Saved figure to: {fig_path}")
        if show:
            plt.show()
        else:
            plt.close(fig)

    def save_data(self, real_data_all, real_data_top20, model_data, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        print("[*] Saving distribution data table...")
        rows = []
        sources = {"Real_All": real_data_all, "Real_Top20": real_data_top20, "PPO_RL": model_data}
        for source_name, data_dict in sources.items():
            for metric_name, values in data_dict.items():
                for value in values:
                    rows.append({"source": source_name, "metric": metric_name, "value": value})
        df = pd.DataFrame(rows)
        csv_path = os.path.join(output_dir, f"{self.model_tag}_distribution_data.csv")
        df.to_csv(csv_path, index=False)
        print(f"[OK] Saved data to: {csv_path}")


def main():
    args = parse_args()
    generator = Figure5GeneratorRL(args.model_path, args.model_tag, args.seed)
    real_all, real_top20, model_data = generator.collect_data(n_samples=args.n_samples)
    generator.save_data(real_all, real_top20, model_data, args.output_dir)
    generator.plot(real_all, real_top20, model_data, args.output_dir, args.show)


if __name__ == "__main__":
    main()
