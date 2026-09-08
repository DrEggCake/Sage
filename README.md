# Sage
An experimental SNN-like system. Built to be integrated into embodied learning and robotics.

This is a continuation in C++ of initial development which is written in Java.
Which is available at https://github.com/DrEggCake/Sage-Java.

## Status

The C++ port now implements the full Sage-Java brain engine:

- `Brain::tick()` — three-phase spiking simulation (transmit → update → learn)
- `setInput()` / `getOutput()` / `resetOutputs()` for sense→think→act
- Reward learning: Hebbian/STDP, ping, leak, eligibility + path traces, dopamine
- Homeostatic plasticity (adaptive thresholds), voltage leak, refractory period
- JSON persistence (save/load trained brains via `BrainSave`)

Plus a **cross-simulator test harness** that runs fixed, deterministic scenarios on
MuJoCo and (via its HTTP harness) OmniSim, comparing trajectories and terminal
metrics. See `tests/README.md`.

## Build

```bash
cmake -B build && cmake --build build -j$(nproc)
./build/sage                 # demo: trains a small brain and saves it
./build/brain_test           # brain engine unit checks
LD_LIBRARY_PATH=$MUJOCO_DIR/lib ./build/cross_sim_test all   # MuJoCo scenarios
```
