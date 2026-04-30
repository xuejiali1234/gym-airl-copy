from __future__ import annotations

import subprocess
from pathlib import Path

from sumo_obs_adapter import SPEC, ensure_sumo_tools


ROOT_DIR = Path(__file__).resolve().parent
NET_DIR = ROOT_DIR / "net"
CFG_DIR = ROOT_DIR / "cfg"
ROUTE_DIR = ROOT_DIR / "routes"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def build_nodes() -> str:
    return f"""
<nodes>
    <node id="main_start" x="0.00" y="0.00" type="priority"/>
    <node id="merge_start" x="{SPEC.merge_start_x_m:.2f}" y="0.00" type="priority"/>
    <node id="merge_end" x="{SPEC.merge_end_x_m:.2f}" y="0.00" type="priority"/>
    <node id="main_end" x="{SPEC.main_end_x_m:.2f}" y="0.00" type="priority"/>
    <node id="ramp_start" x="{SPEC.ramp_start_x_m:.2f}" y="{SPEC.ramp_start_y_m:.2f}" type="priority"/>
</nodes>
"""


def build_edges() -> str:
    lane_width = f'{SPEC.lane_width_m:.4f}'
    return f"""
<edges>
    <edge id="{SPEC.main_in_edge}" from="main_start" to="merge_start" numLanes="2" speed="31.0" priority="3" width="{lane_width}"/>
    <edge id="{SPEC.merge_edge}" from="merge_start" to="merge_end" numLanes="3" speed="28.0" priority="3" width="{lane_width}"/>
    <edge id="{SPEC.main_out_edge}" from="merge_end" to="main_end" numLanes="2" speed="31.0" priority="3" width="{lane_width}"/>
    <edge id="{SPEC.ramp_in_edge}" from="ramp_start" to="merge_start" numLanes="1" speed="24.0" priority="2" width="{lane_width}"/>
</edges>
"""


def build_connections() -> str:
    return f"""
<connections>
    <connection from="{SPEC.main_in_edge}" to="{SPEC.merge_edge}" fromLane="0" toLane="1"/>
    <connection from="{SPEC.main_in_edge}" to="{SPEC.merge_edge}" fromLane="1" toLane="2"/>
    <connection from="{SPEC.ramp_in_edge}" to="{SPEC.merge_edge}" fromLane="0" toLane="0"/>
    <connection from="{SPEC.merge_edge}" to="{SPEC.main_out_edge}" fromLane="1" toLane="0"/>
    <connection from="{SPEC.merge_edge}" to="{SPEC.main_out_edge}" fromLane="2" toLane="1"/>
</connections>
"""


def build_route_placeholder() -> str:
    return f"""
<routes>
    <vType id="bg_car" accel="2.6" decel="4.5" sigma="0.5" length="4.8" minGap="2.0" maxSpeed="31.0"/>
    <vType id="rl_car" accel="3.0" decel="4.5" sigma="0.0" length="4.8" minGap="2.0" maxSpeed="28.0"/>
    <route id="{SPEC.background_route_id}" edges="{SPEC.main_in_edge} {SPEC.merge_edge} {SPEC.main_out_edge}"/>
    <route id="{SPEC.controlled_route_id}" edges="{SPEC.ramp_in_edge} {SPEC.merge_edge} {SPEC.main_out_edge}"/>
</routes>
"""


def build_cfg_template() -> str:
    return """
<configuration>
    <input>
        <net-file value="../net/merge_synthetic.net.xml"/>
        <route-files value="../routes/runtime_routes.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <step-length value="0.1"/>
    </time>
    <processing>
        <collision.action value="warn"/>
        <lanechange.duration value="2.0"/>
    </processing>
</configuration>
"""


def run_netconvert(net_dir: Path) -> Path:
    ensure_sumo_tools()
    net_path = net_dir / "merge_synthetic.net.xml"
    cmd = [
        "netconvert",
        "--node-files",
        str(net_dir / "merge_synthetic.nod.xml"),
        "--edge-files",
        str(net_dir / "merge_synthetic.edg.xml"),
        "--connection-files",
        str(net_dir / "merge_synthetic.con.xml"),
        "--output-file",
        str(net_path),
        "--default.lanewidth",
        f"{SPEC.lane_width_m:.4f}",
        "--no-turnarounds",
        "true",
    ]
    subprocess.run(cmd, check=True)
    return net_path


def main() -> None:
    NET_DIR.mkdir(parents=True, exist_ok=True)
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    ROUTE_DIR.mkdir(parents=True, exist_ok=True)

    write_text(NET_DIR / "merge_synthetic.nod.xml", build_nodes())
    write_text(NET_DIR / "merge_synthetic.edg.xml", build_edges())
    write_text(NET_DIR / "merge_synthetic.con.xml", build_connections())
    write_text(ROUTE_DIR / "runtime_routes.rou.xml", build_route_placeholder())
    write_text(CFG_DIR / "merge_template.sumocfg", build_cfg_template())

    net_path = run_netconvert(NET_DIR)
    print(f"[SUMO PoC] Wrote network files to: {NET_DIR}")
    print(f"[SUMO PoC] Built net: {net_path}")


if __name__ == "__main__":
    main()
