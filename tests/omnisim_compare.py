#!/usr/bin/env python3
"""
OmniSim comparison harness for Sage cross-simulator tests.

Runs the same scenarios on OmniSim via its HTTP/JSON protocol, then compares
trajectories with the MuJoCo CSV results.

Endpoint reference: OmniSim PROTOCOL.md (verified against the repo)
  - GET  /healthz            (ok:true lives here)
  - POST /world/load         (ok:true; {path, wait_s, with_supervisor})
  - POST /sim/step           (no ok; returns sim_time_ms, advanced_to_ms)
  - GET  /scene/tree         (nodes[] each with def/position/is_robot)
  - GET  /sim/contacts

WARNING from the protocol: a single /sim/step can take ~0.5-1 s on a small
Newton world and an over-long request drops the session socket (SUPERVISOR_
RPC_TIMEOUT_S=120, no retry). So this client steps in small batches and reads
positions at the same time cadence as the MuJoCo CSV, never asking for many
steps in one request.

Usage:
    python3 tests/omnisim_compare.py [--harness-url http://127.0.0.1:6789]
"""

import argparse
import csv
import json
import sys
import urllib.request
import urllib.error
from pathlib import Path

HARNESS_URL = "http://127.0.0.1:6789"

# scenario -> (mujoco CSV, omnisim world, dt_s, steps)
SCENARIOS = {
    "cart_pole": {
        "csv": "cart_pole_mujoco.csv",
        "world": "tests/omnisim_worlds/cart_pole.omniworld",
        "dt": 0.02,
        "steps": 500,
    },
    "ball_drop": {
        "csv": "ball_drop_mujoco.csv",
        "world": "tests/omnisim_worlds/ball_drop.omniworld",
        "dt": 0.005,
        "steps": 2000,
    },
    "arm_reach": {
        "csv": "arm_reach_mujoco.csv",
        "world": "tests/omnisim_worlds/arm_reach.omniworld",
        "dt": 0.01,
        "steps": 1000,
    },
}


def http_post(path, data=None):
    url = f"{HARNESS_URL}{path}"
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, method="POST")
    if body:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=150) as resp:
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
        with urllib.request.urlopen(url, timeout=150) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"  GET error: {e}")
        return None


def check_harness():
    result = http_get("/healthz")
    if result and result.get("ok"):
        print(f"  Harness alive, uptime: {result.get('uptime_s', '?')}s")
        return True
    print("  Harness not reachable at", HARNESS_URL)
    return False


def load_world(path):
    result = http_post("/world/load", {
        "path": path,
        "wait_s": 30.0,
        "with_supervisor": True,
    })
    if result and result.get("ok"):
        print(f"  World loaded: {result.get('world')} (load_ms={result.get('load_ms')})")
        return True
    print(f"  Load failed: {result}")
    return False


def sample_position():
    """Read the first dynamic body's world position from /scene/tree."""
    scene = http_get("/scene/tree")
    if scene and "nodes" in scene:
        for node in scene.get("nodes", []):
            pos = node.get("position")
            if pos is not None:
                return [float(p) for p in pos[:3]]
    return None


def run_omnisim_scenario(name, cfg):
    print(f"\n  Running OmniSim scenario: {name}")

    if not load_world(cfg["world"]):
        return None

    dt = cfg["dt"]
    total = cfg["steps"]
    trajectory = []

    # Step in small batches; walk the same number of basic timesteps as the
    # MuJoCo run, sampling one position per basic step would be too chatty, so
    # sample once per batch at the matching sim time.
    batch = 10
    done = 0
    while done < total:
        n = min(batch, total - done)
        step_result = http_post("/sim/step", {"steps": n})
        if not step_result:
            print("  STEP FAILED; stopping scenario")
            break
        done += n

        pos = sample_position()
        if pos is None:
            print("  Could not read position from /scene/tree; stopping scenario")
            break

        trajectory.append({
            "time": done * dt,
            "sim_time_ms": step_result.get("sim_time_ms"),
            "x": pos[0],
            "y": pos[1],
            "z": pos[2],
        })
        if done % 200 < batch:
            print(f"    step {done}/{total}  pos=({pos[0]:.4f},{pos[1]:.4f},{pos[2]:.4f})")

    return trajectory


def load_mujoco_csv(filename):
    path = Path("tests/results") / filename
    if not path.exists():
        print(f"  MuJoCo CSV not found: {path}")
        return []

    traj = []
    with open(path) as f:
        for row in csv.DictReader(f):
            traj.append({
                "time": float(row["time"]),
                "x": float(row["x"]),
                "y": float(row["y"]),
                "z": float(row["z"]),
            })
    return traj


def sample_mujoco(traj, time):
    """Nearest-neighbour sample of the MuJoCo trajectory at a given time."""
    best = min(traj, key=lambda p: abs(p["time"] - time))
    return best


def compare_trajectories(mujo_traj, omni_traj, name):
    print(f"\n  === Comparison: {name} ===")

    if not mujo_traj:
        print("  No MuJoCo trajectory to compare")
        return
    if not omni_traj:
        print("  No OmniSim trajectory to compare")
        return

    errors = []
    for om in omni_traj:
        mj = sample_mujoco(mujo_traj, om["time"])
        err = ((mj["x"] - om["x"]) ** 2 +
               (mj["y"] - om["y"]) ** 2 +
               (mj["z"] - om["z"]) ** 2) ** 0.5
        errors.append(err)

    avg_err = sum(errors) / len(errors) if errors else 0.0
    max_err = max(errors) if errors else 0.0

    print(f"  Samples compared: {len(errors)}")
    print(f"  Avg position error (L2): {avg_err:.6f}")
    print(f"  Max position error (L2): {max_err:.6f}")

    threshold = 0.05
    match = avg_err < threshold
    print(f"  Match (threshold {threshold}): {'YES' if match else 'MISMATCH'}")

    if not match:
        print(f"  *** FIRST MISMATCH: avg={avg_err:.6f} ***")
        print("  Indicates physics-engine divergence between MuJoCo and "
              "OmniSim (Newton/MuJoCo-solver). First setup assumption to "
              "reconcile: there is no shared scene format; MJCF vs .omniworld "
              "differs and friction/damping map imperfectly.")


def set_base_url(url):
    global HARNESS_URL
    HARNESS_URL = url


def main():
    parser = argparse.ArgumentParser(description="OmniSim cross-simulator comparison")
    parser.add_argument("--harness-url", default=HARNESS_URL)
    parser.add_argument("--scenario", default="all",
                        choices=["all", "cart_pole", "ball_drop", "arm_reach"])
    args = parser.parse_args()

    set_base_url(args.harness_url)

    print("=" * 50)
    print("  Sage Cross-Simulator Comparison")
    print("  OmniSim backend (HTTP/JSON)")
    print("=" * 50)

    scope = list(SCENARIOS) if args.scenario == "all" else [args.scenario]

    if not check_harness():
        print("\nOmniSim harness not available; showing MuJoCo-only results.")
        for name in scope:
            csv_path = Path("tests/results") / SCENARIOS[name]["csv"]
            if csv_path.exists():
                print(f"  {name}: {len(load_mujoco_csv(SCENARIOS[name]['csv']))} MuJoCo samples")
            else:
                print(f"  {name}: no MuJoCo results")
        return

    for name in scope:
        cfg = SCENARIOS[name]
        omni_traj = run_omnisim_scenario(name, cfg)
        mujo_traj = load_mujoco_csv(cfg["csv"])

        if omni_traj:
            out_path = Path("tests/results") / f"{name}_omnisim.csv"
            with open(out_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["time", "sim_time_ms", "x", "y", "z"])
                writer.writeheader()
                writer.writerows(omni_traj)
            print(f"  Wrote {out_path}")

        compare_trajectories(mujo_traj, omni_traj, name)

    print("\n" + "=" * 50)
    print("  Cross-simulator comparison complete.")
    print("  Results in tests/results/")
    print("=" * 50)


if __name__ == "__main__":
    main()
