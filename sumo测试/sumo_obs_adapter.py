from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from configs.config import Config


FT_PER_M = 3.28084
M_PER_FT = 0.3048


@dataclass(frozen=True)
class SumoMergeSpec:
    lane_width_ft: float = 12.0
    lane_width_m: float = lane_width_ft * M_PER_FT
    main_in_length_m: float = 180.0
    merge_length_m: float = 140.0
    main_out_length_m: float = 280.0
    ramp_start_x_m: float = 110.0
    ramp_start_y_m: float = -42.0

    main_in_edge: str = "main_in"
    ramp_in_edge: str = "ramp_in"
    merge_edge: str = "merge_zone"
    main_out_edge: str = "main_out"

    controlled_vehicle_id: str = "ramp_rl_0"
    background_route_id: str = "main_bg_route"
    controlled_route_id: str = "ramp_route"

    target_lane_index_main: int = 0
    left_lane_index_main: int = 1
    aux_lane_index_merge: int = 0
    target_lane_index_merge: int = 1
    left_lane_index_merge: int = 2

    target_lane_center_ft: float = 54.0
    left_lane_center_ft: float = 66.0
    aux_lane_center_ft: float = 66.0

    default_goal_y_margin_ft: float = 120.0

    @property
    def merge_start_x_m(self) -> float:
        return self.main_in_length_m

    @property
    def merge_end_x_m(self) -> float:
        return self.merge_start_x_m + self.merge_length_m

    @property
    def main_end_x_m(self) -> float:
        return self.merge_end_x_m + self.main_out_length_m

    @property
    def ramp_goal_x_ft(self) -> float:
        return self.target_lane_center_ft

    @property
    def ramp_goal_y_ft(self) -> float:
        return self.main_end_x_m * FT_PER_M - self.default_goal_y_margin_ft


SPEC = SumoMergeSpec()


def ensure_sumo_tools() -> Path:
    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home:
        tools = Path(sumo_home) / "tools"
        if tools.exists():
            if str(tools) not in sys.path:
                sys.path.append(str(tools))
            return tools

    sumo_bin = shutil.which("sumo")
    if not sumo_bin:
        raise FileNotFoundError("SUMO executable not found in PATH and SUMO_HOME is unset.")

    tools = Path(sumo_bin).resolve().parents[1] / "tools"
    if not tools.exists():
        raise FileNotFoundError(f"SUMO tools directory not found: {tools}")
    if str(tools) not in sys.path:
        sys.path.append(str(tools))
    os.environ.setdefault("SUMO_HOME", str(tools.parent))
    return tools


def load_training_stats(device: str = "cpu") -> Tuple[np.ndarray, np.ndarray]:
    from evaluation.performance_evaluate_transfer import build_default_stats_paths, load_stats_dataset

    dataset, (expert_mean, expert_std) = load_stats_dataset(build_default_stats_paths(), device=device)
    _ = dataset
    return expert_mean.astype(np.float32), expert_std.astype(np.float32)


def lane_role(edge_id: str, lane_index: int) -> str:
    if edge_id == SPEC.main_in_edge:
        if lane_index == SPEC.target_lane_index_main:
            return "target"
        if lane_index == SPEC.left_lane_index_main:
            return "left"
    elif edge_id == SPEC.merge_edge:
        if lane_index == SPEC.aux_lane_index_merge:
            return "aux"
        if lane_index == SPEC.target_lane_index_merge:
            return "target"
        if lane_index == SPEC.left_lane_index_merge:
            return "left"
    elif edge_id == SPEC.main_out_edge:
        if lane_index == SPEC.target_lane_index_main:
            return "target"
        if lane_index == SPEC.left_lane_index_main:
            return "left"
    elif edge_id == SPEC.ramp_in_edge:
        return "aux"
    return "other"


def pseudo_x_ft(edge_id: str, lane_index: int, lane_pos_m: float, lane_length_m: float) -> float:
    role = lane_role(edge_id, lane_index)
    if role == "target":
        return SPEC.target_lane_center_ft
    if role == "left":
        return SPEC.left_lane_center_ft
    if role == "aux":
        if edge_id == SPEC.ramp_in_edge:
            start_ft = SPEC.aux_lane_center_ft + 6.0
            end_ft = SPEC.aux_lane_center_ft
            ratio = np.clip(lane_pos_m / max(lane_length_m, 1e-6), 0.0, 1.0)
            return float(start_ft + (end_ft - start_ft) * ratio)
        if edge_id == SPEC.merge_edge:
            start_ft = SPEC.aux_lane_center_ft
            end_ft = SPEC.target_lane_center_ft + 1.5
            ratio = np.clip(lane_pos_m / max(lane_length_m, 1e-6), 0.0, 1.0)
            return float(start_ft + (end_ft - start_ft) * ratio)
        return SPEC.aux_lane_center_ft
    return SPEC.left_lane_center_ft


def normalize_goal(goal_xy_ft: np.ndarray, expert_mean: np.ndarray, expert_std: np.ndarray) -> np.ndarray:
    goal_mean = expert_mean[0:2]
    goal_std = expert_std[0:2]
    return ((goal_xy_ft - goal_mean) / goal_std).astype(np.float32)


class SumoObsAdapter:
    def __init__(self, traci_module, expert_mean: np.ndarray, expert_std: np.ndarray):
        self.traci = traci_module
        self.cfg = Config()
        self.expert_mean = expert_mean.astype(np.float32)
        self.expert_std = expert_std.astype(np.float32)
        self.prev_xy_ft: Dict[str, Tuple[float, float]] = {}
        self.goal_xy_ft = np.array([SPEC.ramp_goal_x_ft, SPEC.ramp_goal_y_ft], dtype=np.float32)
        self.goal_norm = normalize_goal(self.goal_xy_ft, self.expert_mean, self.expert_std)

    def _vehicle_state(self, veh_id: str) -> Optional[Dict[str, float]]:
        if veh_id not in self.traci.vehicle.getIDList():
            return None
        edge_id = self.traci.vehicle.getRoadID(veh_id)
        lane_index = self.traci.vehicle.getLaneIndex(veh_id)
        lane_id = self.traci.vehicle.getLaneID(veh_id)
        lane_pos_m = self.traci.vehicle.getLanePosition(veh_id)
        lane_length_m = self.traci.lane.getLength(lane_id)
        pos_x_m, pos_y_m = self.traci.vehicle.getPosition(veh_id)
        x_ft = pseudo_x_ft(edge_id, lane_index, lane_pos_m, lane_length_m)
        y_ft = pos_x_m * FT_PER_M
        prev = self.prev_xy_ft.get(veh_id)
        if prev is None:
            vx_ft = 0.0
            vy_ft = self.traci.vehicle.getSpeed(veh_id) * FT_PER_M
        else:
            vx_ft = (x_ft - prev[0]) / self.cfg.DT
            vy_ft = (y_ft - prev[1]) / self.cfg.DT
        self.prev_xy_ft[veh_id] = (x_ft, y_ft)

        return {
            "veh_id": veh_id,
            "edge_id": edge_id,
            "lane_index": lane_index,
            "lane_role": lane_role(edge_id, lane_index),
            "x_ft": x_ft,
            "y_ft": y_ft,
            "vx_ft": vx_ft,
            "vy_ft": vy_ft,
            "lane_pos_m": lane_pos_m,
            "lane_length_m": lane_length_m,
            "speed_mps": self.traci.vehicle.getSpeed(veh_id),
        }

    def _surround_candidates(self, ego_id: str) -> List[Dict[str, float]]:
        states = []
        for veh_id in self.traci.vehicle.getIDList():
            if veh_id == ego_id:
                continue
            state = self._vehicle_state(veh_id)
            if state is not None:
                states.append(state)
        return states

    @staticmethod
    def _pick_lead(ego_state: Dict[str, float], candidates: List[Dict[str, float]], role: str) -> Optional[Dict[str, float]]:
        ahead = [c for c in candidates if c["lane_role"] == role and c["y_ft"] > ego_state["y_ft"]]
        if not ahead:
            return None
        return min(ahead, key=lambda item: item["y_ft"] - ego_state["y_ft"])

    @staticmethod
    def _pick_follow(ego_state: Dict[str, float], candidates: List[Dict[str, float]], role: str) -> Optional[Dict[str, float]]:
        behind = [c for c in candidates if c["lane_role"] == role and c["y_ft"] < ego_state["y_ft"]]
        if not behind:
            return None
        return max(behind, key=lambda item: item["y_ft"])

    @staticmethod
    def _rel_block(ego_state: Dict[str, float], target: Optional[Dict[str, float]]) -> np.ndarray:
        if target is None:
            return np.zeros(4, dtype=np.float32)
        return np.array(
            [
                target["x_ft"] - ego_state["x_ft"],
                target["y_ft"] - ego_state["y_ft"],
                target["vx_ft"] - ego_state["vx_ft"],
                target["vy_ft"] - ego_state["vy_ft"],
            ],
            dtype=np.float32,
        )

    def build_observation(self, ego_id: str) -> Tuple[np.ndarray, Dict[str, object]]:
        ego_state = self._vehicle_state(ego_id)
        if ego_state is None:
            raise ValueError(f"Controlled vehicle not found in SUMO: {ego_id}")

        candidates = self._surround_candidates(ego_id)
        l6_lead = self._pick_lead(ego_state, candidates, "left")
        l5_lead = self._pick_lead(ego_state, candidates, "target")
        l5_follow = self._pick_follow(ego_state, candidates, "target")

        raw_state = np.concatenate(
            [
                np.array(
                    [ego_state["x_ft"], ego_state["y_ft"], ego_state["vx_ft"], ego_state["vy_ft"]],
                    dtype=np.float32,
                ),
                self._rel_block(ego_state, l6_lead),
                self._rel_block(ego_state, l5_lead),
                self._rel_block(ego_state, l5_follow),
            ],
            axis=0,
        )
        normalized_state = (raw_state - self.expert_mean) / self.expert_std
        if l6_lead is None:
            normalized_state[4:8] = 0.0
        if l5_lead is None:
            normalized_state[8:12] = 0.0
        if l5_follow is None:
            normalized_state[12:16] = 0.0

        obs = np.concatenate([normalized_state.astype(np.float32), self.goal_norm], axis=0)
        meta = {
            "ego_state": ego_state,
            "l6_lead": l6_lead,
            "l5_lead": l5_lead,
            "l5_follow": l5_follow,
            "goal_xy_ft": self.goal_xy_ft.copy(),
        }
        return obs.astype(np.float32), meta

    def build_goal_info(self) -> Dict[str, float]:
        return {
            "goal_x_ft": float(self.goal_xy_ft[0]),
            "goal_y_ft": float(self.goal_xy_ft[1]),
        }
