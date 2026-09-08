#!/usr/bin/env python3
"""
OmniSim comparison harness for Sage cross-simulator tests.

Runs the same scenarios on OmniSim via its HTTP/JSON protocol,
then compares trajectories with the MuJoCo CSV results.

Usage:
    python3 tests/omnisim_compare.py [--harness-url http://127.0.0.1:6789]
"""

import argparse
import csv
import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path


HARNESS_URL = "http://127.0.0.1:6789"


def http_post(path, data=None):
    url = f"{HARNESS_URL}{path}"
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, method="POST")
    if body:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.URLError as e:
        print(f"  HTTP error: {e}")
        return None
    except Exception as e:
        print(f"  Request failed: {e}")
        return None


def http_get(path):
    url = f"{HARNESS_URL}{path}"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"  GET error: {e}")
        return None


def check_harness():
    result = http_get("/healthz")
    if result and result.get("ok"):
        print(f"  Harness alive, uptime: {result.get('uptime_s', '?')}s")
        return True
    print("  Harness not reachable!")
    return False


def load_world(path):
    result = http_post("/world/load", {
        "path": path,
        "wait_s": 30.0,
        "with_supervisor": True,
        "light": True
    })
    if result and result.get("ok"):
        print(f"  World loaded: {result.get('world')}")
        return True
    print(f"  Load failed: {result}")
    return False


def step_sim(n_steps=1):
    results = []
    for _ in range(n_steps):
        result = http_post("/sim/step", {"steps": 1})
        results.append(result)
        if not result:
            break
    return results


def get_scene_tree():
    return http_get("/scene/tree")


def get_contacts():
    return http_get("/sim/contacts")


def read_joints(robot_def):
    return http_post("/get_robot_state", {"robot_id": robot_def})


def run_omnisim_scenario(scenario_name, omnisim_world_path, n_steps=100):
    print(f"\n  Running OmniSim scenario: {scenario_name}")

    if not load_world(omnisim_world_path):
        return None

    trajectory = []
    for i in range(n_steps):
        step_result = step_sim(1)
        if not step_result:
            break

        state = http_get("/sim/state")
        scene = get_scene_tree()

        point = {
            "time": i * 0.02,
            "sim_time_ms": step_result[0].get("sim_time_ms", 0) if step_result else 0,
        }

        if scene and "nodes" in scene:
            for node in scene["nodes"]:
                if node.get("is_robot") or "ball" in node.get("def", "").lower() or "cart" in node.get("def", "").lower():
                    pos = node.get("position", [0, 0, 0])
                    point["position"] = pos
                    break

        trajectory.append(point)

        if i % 200 == 0:
            print(f"    step {i}/{n_steps}")

    return trajectory


def load_mujoco_csv(filename):
    trajectory = []
    path = Path("tests/results") / filename
    if not path.exists():
        print(f"  MuJoCo CSV not found: {path}")
        return []

    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            trajectory.append({
                "time": float(row["time"]),
                "x": float(row["x"]),
                "y": float(row["y"]),
                "z": float(row["z"]),
            })
    return trajectory


def compare_trajectories(mujo_traj, omni_traj, scenario_name):
    print(f"\n  === Comparison: {scenario_name} ===")

    if not mujo_traj:
        print("  No MuJoCo trajectory to compare")
        return
    if not omni_traj:
        print("  No OmniSim trajectory to compare")
        return

    min_len = min(len(mujo_traj), len(omni_traj))

    max_pos_err = 0.0
    pos_errors = []

    for i in range(min_len):
        mj = mujo_traj[i]
        om = omni_traj[i]

        mj_pos = (mj.get("x", 0), mj.get("y", 0), mj.get("z", 0))
        om_pos = tuple(om.get("position", [0, 0, 0])[:3]) if "position" in om else (0, 0, 0)

        err = sum((a - b) ** 2 for a, b in zip(mj_pos, om_pos)) ** 0.5
        pos_errors.append(err)
        if err > max_pos_err:
            max_pos_err = err

    avg_err = sum(pos_errors) / len(pos_errors) if pos_errors else 0

    print(f"  Samples compared: {min_len}")
    print(f"  Avg position error (L2): {avg_err:.6f}")
    print(f"  Max position error (L2): {max_pos_err:.6f}")

    threshold = 0.05
    match = avg_err < threshold
    print(f"  Match (threshold {threshold}): {'YES' if match else 'MISMATCH'}")

    if not match:
        print(f"  *** FIRST MISMATCH: avg={avg_err:.6f} ***")
        print(f"  This indicates physics engine differences between MuJoCo and OmniSim (Newton).")


def main():
    parser = argparse.ArgumentParser(description="OmniSim cross-simulator comparison")
    parser.add_argument("--harness-url", default=HARNESS_URL)
    parser.add_argument("--scenario", default="all", choices=["all", "cart_pole", "ball_drop", "arm_reach"])
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()

    global HARNESS_URL
    HARNESS_URL = args.harness_url

    print("=" * 50)
    print("  Sage Cross-Simulator Comparison")
    print("  OmniSim backend (HTTP/JSON)")
    print("=" * 50)

    print("\nChecking OmniSim harness...")
    if not check_harness():
        print("\nOmniSim harness not available.")
        print("Start it with: python -m omnisim harness")
        print("\nComparing with MuJoCo-only results...\n")

        for scenario in ["cart_pole", "ball_drop", "arm_reach"]:
            if args.scenario in ("all", scenario):
                csv_path = Path(f"tests/results/{scenario}_mujoco.csv")
                if csv_path.exists():
                    traj = load_mujoco_csv(f"{scenario}_mujoco.csv")
                    print(f"  {scenario}: {len(traj)} MuJoCo samples loaded")
                else:
                    print(f"  {scenario}: no MuJoCo results found")
        return

    omnisim_scenarios = {
        "cart_pole": "projects/samples/demos/worlds/physics/cart_pole.omniworld",
        "ball_drop": "projects/samples/demos/worlds/physics/ball_drop.omniworld",
        "arm_reach": "projects/samples/demos/worlds/physics/arm_reach.omniworld",
    }

    for scenario, world_path in omnisim_scenarios.items():
        if args.scenario in ("all", scenario):
            omni_traj = run_omnisim_scenario(scenario, world_path, args.steps)
            mujo_traj = load_mujoco_csv(f"{scenario}_mujoco.csv")

            if omni_traj:
                out_path = Path(f"tests/results/{scenario}_omnisim.csv")
                with open(out_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=["time", "sim_time_ms", "x", "y", "z"])
                    writer.writeheader()
                    writer.writerows(omni_traj)

            compare_trajectories(mujo_traj, omni_traj, scenario)

    print("\n" + "=" * 50)
    print("  Cross-simulator comparison complete.")
    print("  Results in tests/results/")
    print("=" * 50)


if __name__ == "__main__":
    main()
