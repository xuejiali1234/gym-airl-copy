from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from stable_baselines3 import PPO

ROOT_DIR = Path(__file__).resolve().parents[1]
POC_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from configs.config import Config
from evaluation.performance_evaluate_transfer import build_default_model_path
from sumo_obs_adapter import SPEC, SumoObsAdapter, ensure_sumo_tools, load_training_stats


DEFAULT_LANECHANGE_MODE = 1621
DEFAULT_SPEED_MODE = 31


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SUMO merge PoC with alternating RL/SUMO ramp vehicles.")
    parser.add_argument("--gui", action="store_true", help="Use sumo-gui instead of headless sumo.")
    parser.add_argument("--steps", type=int, default=900, help="Maximum SUMO steps to simulate.")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--model-path", default=build_default_model_path())
    parser.add_argument("--mainline-flow-rate", type=float, default=2200.0, help="Total mainline veh/h across two lanes.")
    parser.add_argument("--ramp-depart-speed", type=float, default=18.0, help="Ramp vehicle depart speed in m/s.")
    parser.add_argument("--ramp-vehicle-count", type=int, default=6, help="Total ramp vehicles to spawn sequentially.")
    parser.add_argument("--ramp-headway-seconds", type=float, default=7.0, help="Depart headway between consecutive ramp vehicles.")
    parser.add_argument("--output-dir", default="", help="Output directory for logs.")
    return parser.parse_args()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def make_output_dir(base: Optional[str]) -> Path:
    if base:
        output_dir = Path(base).resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (POC_DIR / "outputs" / f"merge_policy_run_{stamp}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def build_ramp_plan(count: int, headway_seconds: float, depart_speed: float) -> List[Dict[str, object]]:
    plan: List[Dict[str, object]] = []
    base_depart = 8.0
    for idx in range(count):
        controller_type = "rl" if idx % 2 == 0 else "sumo"
        veh_id = f"ramp_{controller_type}_{idx:02d}"
        color = "1,0.2,0.2" if controller_type == "rl" else "0.2,0.2,1"
        plan.append(
            {
                "veh_id": veh_id,
                "controller_type": controller_type,
                "depart": base_depart + idx * headway_seconds,
                "depart_speed": depart_speed,
                "color": color,
            }
        )
    return plan


def build_runtime_routes(args: argparse.Namespace, ramp_plan: List[Dict[str, object]]) -> str:
    begin_main = 0.0
    end_main = max(args.steps * Config.DT, 1.0)
    per_lane = max(args.mainline_flow_rate / 2.0, 120.0)
    ramp_vehicles = []
    for item in ramp_plan:
        vtype = "rl_car" if item["controller_type"] == "rl" else "ramp_human"
        ramp_vehicles.append(
            f'''    <vehicle id="{item["veh_id"]}" type="{vtype}" route="{SPEC.controlled_route_id}"
             depart="{item["depart"]:.1f}" departLane="0" departPos="0" departSpeed="{item["depart_speed"]:.2f}" color="{item["color"]}"/>'''
        )

    return f"""
<routes>
    <vType id="bg_car" accel="2.6" decel="4.5" sigma="0.5" tau="1.0" length="4.8" minGap="2.0" maxSpeed="31.0"/>
    <vType id="ramp_human" accel="2.4" decel="4.5" sigma="0.5" tau="1.1" length="4.8" minGap="2.0" maxSpeed="26.0"/>
    <vType id="rl_car" accel="3.0" decel="4.5" sigma="0.0" tau="1.0" length="4.8" minGap="2.0" maxSpeed="28.0"/>

    <route id="{SPEC.background_route_id}" edges="{SPEC.main_in_edge} {SPEC.merge_edge} {SPEC.main_out_edge}"/>
    <route id="{SPEC.controlled_route_id}" edges="{SPEC.ramp_in_edge} {SPEC.merge_edge} {SPEC.main_out_edge}"/>

    <flow id="bg_target_flow" type="bg_car" route="{SPEC.background_route_id}"
          begin="{begin_main:.1f}" end="{end_main:.1f}" vehsPerHour="{per_lane:.1f}" departLane="0" departSpeed="max"/>
    <flow id="bg_left_flow" type="bg_car" route="{SPEC.background_route_id}"
          begin="{begin_main:.1f}" end="{end_main:.1f}" vehsPerHour="{per_lane:.1f}" departLane="1" departSpeed="max"/>

{chr(10).join(ramp_vehicles)}
</routes>
"""


def build_runtime_cfg(net_path: Path, routes_path: Path, output_dir: Path) -> str:
    tripinfo_path = output_dir / "tripinfo.xml"
    fcd_path = output_dir / "fcd.xml"
    return f"""
<configuration>
    <input>
        <net-file value="{net_path.as_posix()}"/>
        <route-files value="{routes_path.as_posix()}"/>
    </input>
    <time>
        <begin value="0"/>
        <step-length value="{Config.DT}"/>
    </time>
    <processing>
        <collision.action value="warn"/>
        <lanechange.duration value="2.0"/>
    </processing>
    <output>
        <tripinfo-output value="{tripinfo_path.as_posix()}"/>
        <fcd-output value="{fcd_path.as_posix()}"/>
    </output>
</configuration>
"""


def choose_sumo_binary(use_gui: bool) -> str:
    binary = shutil.which("sumo-gui" if use_gui else "sumo")
    if not binary:
        raise FileNotFoundError(f"SUMO binary not found for gui={use_gui}")
    return binary


def setup_traci():
    ensure_sumo_tools()
    import traci  # type: ignore

    return traci


def lane_change_request(action_x: float, ego_state: Dict[str, float]) -> Optional[int]:
    if ego_state["edge_id"] != SPEC.merge_edge:
        return None
    if ego_state["lane_index"] != SPEC.aux_lane_index_merge:
        return None
    if action_x < -0.12:
        return SPEC.target_lane_index_merge
    return None


def apply_policy_control(traci, veh_id: str, action: np.ndarray, ego_state: Dict[str, float], cfg: Config) -> Dict[str, object]:
    current_speed = traci.vehicle.getSpeed(veh_id)
    acc_mps2 = float(action[1]) * cfg.PHYS_ACC_MAX * 0.3048
    desired_speed = float(np.clip(current_speed + acc_mps2 * cfg.DT, 0.0, 31.0))
    traci.vehicle.setSpeed(veh_id, desired_speed)

    requested_lane = lane_change_request(float(action[0]), ego_state)
    if requested_lane is not None:
        traci.vehicle.changeLane(veh_id, requested_lane, 2.0)

    return {
        "acc_mps2": acc_mps2,
        "desired_speed_mps": desired_speed,
        "requested_lane": requested_lane,
    }


def set_controlled_modes(traci, veh_id: str) -> None:
    traci.vehicle.setSpeedMode(veh_id, DEFAULT_SPEED_MODE)
    traci.vehicle.setLaneChangeMode(veh_id, 0)


def restore_sumo_modes(traci, veh_id: str) -> None:
    traci.vehicle.setSpeedMode(veh_id, DEFAULT_SPEED_MODE)
    traci.vehicle.setLaneChangeMode(veh_id, DEFAULT_LANECHANGE_MODE)
    traci.vehicle.setSpeed(veh_id, -1)


def save_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def init_vehicle_records(ramp_plan: List[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    records: Dict[str, Dict[str, object]] = {}
    for item in ramp_plan:
        veh_id = str(item["veh_id"])
        records[veh_id] = {
            "vehicle_id": veh_id,
            "controller_type": item["controller_type"],
            "depart_time_s": item["depart"],
            "first_seen_step": None,
            "merge_target_first_step": None,
            "handover_step": None,
            "handover_success": False,
            "collision": False,
            "collision_step": None,
            "arrived": False,
            "arrival_step": None,
            "controlled_present_ever": False,
            "final_edge_id": "",
            "final_lane_index": -1,
            "final_control_mode": "",
            "merge_success": False,
            "steps_logged": 0,
        }
    return records


def summarize_compare(records: Dict[str, Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    groups = {}
    for record in records.values():
        groups.setdefault(record["controller_type"], []).append(record)

    for controller_type, items in groups.items():
        total = len(items)
        collisions = sum(int(bool(item["collision"])) for item in items)
        merges = sum(int(bool(item["merge_success"])) for item in items)
        arrivals = sum(int(bool(item["arrived"])) for item in items)
        handovers = sum(int(bool(item["handover_success"])) for item in items)
        rows.append(
            {
                "controller_type": controller_type,
                "vehicle_count": total,
                "merge_success_rate": merges / total if total else 0.0,
                "arrival_rate": arrivals / total if total else 0.0,
                "collision_rate": collisions / total if total else 0.0,
                "handover_rate": handovers / total if total else 0.0,
            }
        )
    return rows


def run_episode(args: argparse.Namespace) -> Path:
    from build_merge_network import main as build_network_main

    build_network_main()

    ramp_plan = build_ramp_plan(args.ramp_vehicle_count, args.ramp_headway_seconds, args.ramp_depart_speed)
    rl_vehicle_ids = {str(item["veh_id"]) for item in ramp_plan if item["controller_type"] == "rl"}
    ramp_vehicle_ids = {str(item["veh_id"]) for item in ramp_plan}

    output_dir = make_output_dir(args.output_dir)
    route_path = output_dir / "runtime_routes.rou.xml"
    cfg_path = output_dir / "runtime.sumocfg"
    write_text(route_path, build_runtime_routes(args, ramp_plan))
    net_path = POC_DIR / "net" / "merge_synthetic.net.xml"
    write_text(cfg_path, build_runtime_cfg(net_path, route_path, output_dir))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    expert_mean, expert_std = load_training_stats(device=str(device))
    model = PPO.load(args.model_path, device=device)
    cfg = Config()

    traci = setup_traci()
    sumo_cmd = [
        choose_sumo_binary(args.gui),
        "-c",
        str(cfg_path),
        "--seed",
        str(args.seed),
        "--no-step-log",
        "true",
        "--time-to-teleport",
        "-1",
    ]
    traci.start(sumo_cmd)

    adapter = SumoObsAdapter(traci, expert_mean, expert_std)
    step_rows: List[Dict[str, object]] = []
    traj_rows: List[Dict[str, object]] = []
    vehicle_records = init_vehicle_records(ramp_plan)
    merge_streaks = {veh_id: 0 for veh_id in ramp_vehicle_ids}
    remaining_ids = set(ramp_vehicle_ids)

    try:
        for step in range(args.steps):
            traci.simulationStep()
            sim_time = traci.simulation.getTime()
            active_ids = set(traci.vehicle.getIDList())
            collided_ids = set(traci.simulation.getCollidingVehiclesIDList())
            arrived_ids = set(traci.simulation.getArrivedIDList())

            for veh_id in list(ramp_vehicle_ids):
                record = vehicle_records[veh_id]
                if veh_id in collided_ids and not record["collision"]:
                    record["collision"] = True
                    record["collision_step"] = step

                if veh_id in active_ids:
                    record["controlled_present_ever"] = True
                    if record["first_seen_step"] is None:
                        record["first_seen_step"] = step

                    ego_obs, meta = adapter.build_observation(veh_id)
                    ego_state = meta["ego_state"]
                    in_target_post_merge = (
                        ego_state["edge_id"] == SPEC.main_out_edge
                        and ego_state["lane_index"] == SPEC.target_lane_index_main
                    )
                    if in_target_post_merge:
                        merge_streaks[veh_id] += 1
                        if record["merge_target_first_step"] is None:
                            record["merge_target_first_step"] = step
                            record["merge_success"] = True
                    else:
                        merge_streaks[veh_id] = 0

                    control_mode = "sumo_baseline"
                    action = np.zeros(2, dtype=np.float32)
                    control_info = {
                        "desired_speed_mps": traci.vehicle.getSpeed(veh_id),
                        "requested_lane": None,
                    }

                    if veh_id in rl_vehicle_ids:
                        control_mode = "policy"
                        if record["handover_step"] is None and merge_streaks[veh_id] >= 10:
                            restore_sumo_modes(traci, veh_id)
                            record["handover_step"] = step
                            record["handover_success"] = True

                        if record["handover_step"] is None:
                            set_controlled_modes(traci, veh_id)
                            action, _ = model.predict(ego_obs, deterministic=True)
                            action = np.asarray(action, dtype=np.float32).reshape(-1)
                            control_info = apply_policy_control(traci, veh_id, action, ego_state, cfg)
                        else:
                            control_mode = "sumo_after_handover"

                    record["final_edge_id"] = ego_state["edge_id"]
                    record["final_lane_index"] = ego_state["lane_index"]
                    record["final_control_mode"] = control_mode
                    record["steps_logged"] = int(record["steps_logged"]) + 1

                    traj_rows.append(
                        {
                            "vehicle_id": veh_id,
                            "controller_type": record["controller_type"],
                            "step": step,
                            "time_s": sim_time,
                            "edge_id": ego_state["edge_id"],
                            "lane_index": ego_state["lane_index"],
                            "lane_role": ego_state["lane_role"],
                            "x_ft": ego_state["x_ft"],
                            "y_ft": ego_state["y_ft"],
                            "vx_ft": ego_state["vx_ft"],
                            "vy_ft": ego_state["vy_ft"],
                            "speed_mps": ego_state["speed_mps"],
                            "control_mode": control_mode,
                            "action_x": float(action[0]),
                            "action_y": float(action[1]),
                            "requested_lane": control_info["requested_lane"],
                            "desired_speed_mps": control_info["desired_speed_mps"],
                            "collision": record["collision"],
                        }
                    )

                    step_rows.append(
                        {
                            "vehicle_id": veh_id,
                            "controller_type": record["controller_type"],
                            "step": step,
                            "time_s": sim_time,
                            "control_mode": control_mode,
                            "edge_id": ego_state["edge_id"],
                            "lane_index": ego_state["lane_index"],
                            "lane_role": ego_state["lane_role"],
                            "merge_streak": merge_streaks[veh_id],
                            "handover_done": record["handover_success"],
                            "collision": record["collision"],
                            "goal_x_ft": meta["goal_xy_ft"][0],
                            "goal_y_ft": meta["goal_xy_ft"][1],
                            "l6_lead_id": meta["l6_lead"]["veh_id"] if meta["l6_lead"] else "",
                            "l5_lead_id": meta["l5_lead"]["veh_id"] if meta["l5_lead"] else "",
                            "l5_follow_id": meta["l5_follow"]["veh_id"] if meta["l5_follow"] else "",
                            "action_x": float(action[0]),
                            "action_y": float(action[1]),
                            "desired_speed_mps": control_info["desired_speed_mps"],
                            "requested_lane": control_info["requested_lane"],
                        }
                    )

                if veh_id in arrived_ids:
                    record["arrived"] = True
                    if record["arrival_step"] is None:
                        record["arrival_step"] = step
                    remaining_ids.discard(veh_id)

                if veh_id not in active_ids and veh_id not in arrived_ids and record["controlled_present_ever"]:
                    remaining_ids.discard(veh_id)

            if not remaining_ids:
                break

    finally:
        traci.close()

    compare_rows = summarize_compare(vehicle_records)
    summary = {
        "model_path": str(Path(args.model_path).resolve()),
        "net_path": str(net_path.resolve()),
        "route_path": str(route_path.resolve()),
        "cfg_path": str(cfg_path.resolve()),
        "steps_requested": args.steps,
        "steps_executed": max((int(r["step"]) for r in traj_rows), default=-1) + 1,
        "mainline_flow_rate": args.mainline_flow_rate,
        "ramp_depart_speed_mps": args.ramp_depart_speed,
        "ramp_vehicle_count": args.ramp_vehicle_count,
        "ramp_headway_seconds": args.ramp_headway_seconds,
        "rl_vehicle_count": len(rl_vehicle_ids),
        "sumo_vehicle_count": len(ramp_vehicle_ids - rl_vehicle_ids),
        "compare_rows": compare_rows,
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    save_csv(output_dir / "step_log.csv", step_rows)
    save_csv(output_dir / "vehicle_trajectory.csv", traj_rows)
    save_csv(output_dir / "vehicle_summary.csv", list(vehicle_records.values()))
    save_csv(output_dir / "controller_compare.csv", compare_rows)
    return output_dir


def main() -> None:
    args = parse_args()
    output_dir = run_episode(args)
    print(f"[SUMO PoC] Finished. Outputs: {output_dir}")


if __name__ == "__main__":
    main()
