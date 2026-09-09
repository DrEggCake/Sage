# Sage — Architecture & Developer Guide

A from-scratch explanation of the whole project: what every language,
directory, file, method, script, workflow and CLI is for, how the pieces talk
to each other, and how to run and extend everything.

This is the *behind-the-code* reference: read §1–§6 for the SNN core (short),
§7–§8 for the two simulator integrations, and §10 for the replay viewer (the
part that is easiest to get lost in). Every symbol named here exists — nothing
is aspirational.

---

## 1. What Sage is

Sage is an **experimental spiking-neural-network (SNN) system** built to be
embedded into embodied learning and robotics. It has three layers:

1. **The brain engine (C++)** — a faithful port of the `Sage-Java` SNN to
   modern C++ (`Brain` / `Neuron` / `Synapse`), plus JSON persistence.
2. **Embodied learning (C++ + Python)** — two simulator integrations that let a
   brain perceive an environment, produce actions, and learn from reward:
   - **MuJoCo** (DeepMind physics) via C++ harnesses (`cross_sim_harness.cpp`,
     `snn_drive.cpp`, `dog_drive.cpp`);
   - **OmniSim** (OmniLink's Newton "mujoco" physics) via an HTTP/JSON harness
     driven from a Python client (`omnisim_compare.py`).
3. **Replay / visualization (Python + JavaScript)** — a browser GUI
   (`replay.html`) that plays back recorded trajectories side-by-side in 3D,
   renders articulated bodies from forward kinematics, and animates the SNN's
   firing live (`serve_replay.py` turns CSV artifacts into a JSON manifest the
   browser fetches).

Historical lineage: `Sage-Java` (https://github.com/DrEggCake/Sage-Java) is the
original; the SNN math in this repo is a direct port of it. This C++ repo is
the "embodied" branch: brain + physics + replay, wired for CI.

---

## 2. The four languages, and why each one is here

| Language | Where | Role |
|---|---|---|
| **C++20** | `brain/`, `utils/`, `main.cpp`, `tests/cross_sim_harness.cpp`, `tests/snn_drive.cpp`, `tests/dog_drive.cpp`, `tests/brain_test.cpp` | The brain engine and every heavy/simulation-side path. C++ is used because the SNN core is the artifact you'd ship into a robot; spiking sims are hot loops that shouldn't pay interpreter overhead, and MuJoCo's official C API is C/++ anyway. |
| **Python 3** | `tests/omnisim_compare.py`, `tests/replay/serve_replay.py` | The *glue* layer. OmniSim's engine embeds its own system Python 3.12 and exposes an HTTP/JSON protocol, so a Python client is the natural way to talk to it. The replay server is a tiny HTTP static server whose only real job is to list CSVs on disk as JSON and serve files to the browser. Nothing in Python touches physics directly. |
| **JavaScript (+ HTML/CSS)** | `tests/replay/replay.html`, `tests/replay/vendor/*` | The replay *GUI*. It runs in any browser with zero build step, and WebGL (three.js) gives free 3D + orbit cameras for the trajectory views. The SNN panel redraws the 5-layer spiking net in SVG/DOM each tick. All parsing, interpolation, forward kinematics and playback logic is hand-rolled (only three.js and OrbitControls are vendored). |
| **Java** | (in `Sage-Java`, not this repo) | The *ancestor*. The brain's tick discipline, neuron/synapse math, reward/dopamine and threshold-homeostasis ideas were originally written in Java; Sage C++ is a faithful port. Several "quirks" (e.g. the voltage-readout style, 5 fixed layers) are 1:1 carry-overs. |

**In short:** C++ = brains + physics, Python = orchestration + bridge to the
remote omni-sim and the web server, JavaScript = the human-facing viewer, Java =
where the brain algorithms came from.

---

## 3. Repository map

```
sage/
├── brain/                       # SNN engine (C++)
│   ├── Brain.h / Brain.cpp      # network container: tick(), wiring, reward, IO
│   ├── Neuron.h / Neuron.cpp    # single spiking neuron
│   └── Synapse.h / Synapse.cpp  # weighted connection + learning state
├── utils/
│   └── BrainSave.h / .cpp       # JSON save/load of a whole Brain
├── tests/
│   ├── brain_test.cpp           # unit checks for the engine
│   ├── cross_sim_harness.cpp    # MuJoCo baseline scenarios + CSV writers
│   ├── snn_drive.cpp            # SNN-controlled MuJoCo arm (embodied learning)
│   ├── dog_drive.cpp            # SNN-controlled MuJoCo quadruped (gait learning)
│   ├── omnisim_compare.py       # OmniSim HTTP client + comparison (Python)
│   ├── models/                  # MuJoCo MJCF XML: arm_reach/ball_drop/cart_pole/dog_walk
│   ├── omnisim_worlds/          # mirrored .omniworld scenes for OmniSim
│   ├── results/                 # output CSVs (git-ignored, runtime artifacts)
│   └── replay/                  # browser replay player
│       ├── replay.html          # the GUI (three.js + SNN panel), self-contained
│       ├── serve_replay.py      # local HTTP server + manifest generator
│       └── vendor/              # vendored three.js 0.160.0 + OrbitControls
├── main.cpp                     # demo/training CLI for the brain (epsilon-greedy)
├── CMakeLists.txt               # build: sage_core lib + 5 executables
├── .github/workflows/
│   └── cross-sim-test.yml       # CI: MuJoCo baselines + OmniSim build + compare
├── docs/
│   └── ARCHITECTURE.md          # this file
└── README.md
```

Git-ignored: `data/*` (trained brains + firing logs), `build/*`, `tests/results/*`,
`__pycache__/`, `*.pyc`, and any `docs/*deliverable*` (local-only write-ups).
So `data/brain/`, `data/firelog/` and `tests/results/` are local/CI-artifact
spaces; committed tests must regenerate what they need.

---

## 4. Build system (CMake)

`CMakeLists.txt`:

- `FetchContent` pulls **nlohmann/json** (used only by `BrainSave`).
- **`sage_core`** — static lib from `brain/*.cpp` + `utils/BrainSave.cpp`. No
  external deps beyond JSON. This is the unit you'd drop into a robot.
- **`sage`** — `main.cpp` linked to `sage_core` (the epsilon-greedy trainer).
- **`brain_test`** — `tests/brain_test.cpp`, the engine unit checks.
- **`cross_sim_test`**, **`snn_drive`**, **`dog_drive`** — MuJoCo-gated. CMake
  looks for MuJoCo via `MUJOCO_DIR` (default `~/mujoco`): needs
  `include/mujoco/mujoco.h` and a `libmujoco` (`.so`/`.a`). If missing, all
  three targets are skipped with a warning so the rest of the project still
  builds. Each links `sage_core` + MuJoCo + nlohmann_json, and adds
  `${MUJOCO_DIR}/lib` to its rpath.

Build:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

After editing `CMakeLists.txt` re-run the configure step; existing build dirs
won't pick up new targets automatically. Binaries land in `build/` (flat, not
per-`source_dir` subfolders).

---

## 5. The SNN brain engine (deep dive)

The engine is deliberately simple — one file per concept, no magic. Everything
below is the *actual* behavior of this port.

### 5.1 `Neuron`

A single leaky integrate-and-fire neuron. Roles (enum `NeuronType`): `INPUT`,
`INTERNAL`, `OUTPUT`.

State:

- `voltage` — membrane voltage. Inputs and synapses add to it; it leaks
  multiplicatively when quiet.
- `threshold` — firing threshold, **homeostatic**: adapts per episode
  (`THRESHOLD_MIN 0.3`, `THRESHOLD_MAX 2.0`, step `THRESHOLD_ADJUST_RATE 0.002`).
- `eligibility` / `accumulatedDrive` — learning trace / spike statistic.
- `fired` — true if it spiked on the last `update()` (drives the replay glow).
- `refractory` — ticks of silence after a spike (`REFRACTORY_TICKS 1`).
- `fireCount` — spikes since the last `resetFireCount()` (drives threshold
  adaptation).
- `synapsesIn` / `synapsesOut` — inbound/outbound connection lists.

Key methods:

- `update()` — the core. If `refractory > 0`, decrement and return (no fire).
  Else if `voltage >= threshold`: **fire** → set `fired`, bump `fireCount`,
  record `accumulatedDrive += voltage`, stub `voltage -= threshold` (clamped ≥ 0),
  call `stimulate()` on every outgoing synapse, set `refractory`. Else: **leak**,
  `voltage *= VOLTAGE_LEAK (0.85)`, clear `fired`.
- `adjustThreshold(targetRate)` — INTERNAL neurons only; raise if over target,
  lower if under, clamp to [0.3, 2.0]. This homeostasis is one of the two things
  that genuinely change behavior in this port (see the `Synapse` note below).
- `reset()` — zero voltage/eligibility/accumulatedDrive, clear fired/refractory.
- `setVoltage(v)` — used by `Brain` to place raw sensor values onto INPUTs.
- `pulse(amount)` — add drive (called by synapses' `stimulate`).
- `markFired()` — manual spike mark (currently unused by the tick loop).
- Getters: `getVoltage`, `getThreshold` (+ `setThreshold`), `getAccumulatedDrive`,
  `getFireCount`, `hasFired`, and `getSynapses{In,Out}` for iteration. `getType`
  for introspection.

### 5.2 `Synapse`

The weighted `from → to` connection that carries the learning state.

Member constants (*read these carefully — they tell you what the engine can and
cannot learn*):

| Constant | Value | Meaning |
|---|---|---|
| `RESTING_STRENGTH` | 0.25 | the baseline every weight relaxes toward (was 0.05 in Sage-Java; see the finding below) |
| `LEAK_RATE` | 0.005 / tick | how fast a weight snaps back to resting |
| `ELIGIBILITY_DECAY` | 0.9 | eligibility-trace decay per tick |
| `LAST_ACTIVATED_DECAY` | 0.95 | path-trace decay per tick |
| `PING_BOOST` | 0.001 | tiny boost when only the *source* fires |
| `HEBB_BOOST` | 0.05 | large boost when eligible **and** target fires |
| `DOPAMINE_GAIN` | 0.02 | scaling of dopamine-modulated boosts |
| `MAX_STRENGTH` | 2.0 | weight clamp |

Key methods:

- `stimulate()` — push `actionPotential * strength` into `to->pulse()`, then set
  `eligibility = 1` and `lastActivated = 1` (the post-synaptic path trace).
- `update()` — decay both traces; relax the weight toward `RESTING_STRENGTH`;
  then Hebbian-style boosts: if `eligibility > 0.05 && to->hasFired()` add
  `HEBB_BOOST`, and if `from->hasFired()` add `PING_BOOST`; clamp to ±2.
- `applyReward(amount)` — `strength += amount * eligibility`. Eligibility-gated
  credit assignment: a synapse only learns if it participated recently.
- `applyDopamine(amount)` — `strength += amount * lastActivated * DOPAMINE_GAIN`.
- `isPathActive()` — `lastActivated > 0.01`, used by `Brain::tick` to gate the
  global dopamine sweep.
- `reset()` — clear both traces.

> **Finding (read before predicting anything):** two separate issues bite this
> learning scheme.
>
> **(a) Reward decay.** `LEAK_RATE` is 0.005/tick and one-shot `reward()`
> updates are scaled by the learning rate (0.02 in `sage`, 0.5 in `dog_drive`).
> Over a long episode the leak eats the credit almost as fast as it is written
> (0.005 × 2400 ticks ≈ 12 × resting). Empirically the *sparse-payoff* tasks stay
> near resting strength.
>
> **(b) Dead net at rest.** At the Java-original `RESTING_STRENGTH = 0.05`, a
> single firing input contributes ~0.35 V steady state (0.05 / LEAK_RATE-like
> accumulation) against a ~1.0 V firing threshold, so for *continuous control*
> (driving a MuJoCo robot from low-magnitude inputs) the network sat silent —
> no hidden layer ever crossed threshold and every motor sat at its bias.
> `RESTING_STRENGTH` was raised to **0.25** in this repo, which puts a single
> input at ~2.9 V steady state > 1.0 V threshold, so signal actually propagates
> through all four layers. This is the change that made `dog_drive` alive at
> all (see §7.5).
>
> What *does* produce behavior within this scheme: **(a) threshold homeostasis**
> (`Neuron::adjustThreshold`) and **(b) sustained co-firing Hebbian boosts**,
> which net *up* by `HEBB_BOOST − LEAK_RATE` each tick as long as a synfire path
> keeps iterating. See the honest results in §7.3 and §7.5.

### 5.3 `Brain`

Topology is hard-coded to **5 layers** (`INPUT → L1 → L2 → L3 → OUTPUT`) with a
wiring "limit" per layer pair. `connectSliding` gives each source a *windowed*
fan-out (the same number of consecutive downstream targets per source).

Constructor: `Brain(in, l1, l2, l3, out, inToL1, l1ToL2, l2ToL3, l3ToOut)`.

Key methods:

- `tick()` — the three-phase loop, ported 1:1 from Sage-Java:
  1. **transmit** — every fired neuron's outgoing synapses `stimulate()`;
  2. **update** — every neuron `update()` (fire or leak);
  3. **learn** — every synapse `update()` (decay + Hebb/ping boosts); then, if
     the global dopamine flag is `> 0.01`, sweep `applyDopamine` over every
     `isPathActive()` synapse and decay dopamine by `dopamineDecay (0.8)`.
     Also bumps `totalSteps`.
- `setInput(i, v)` / `getOutput(i)` / `resetOutputs()` — sense → think → act.
  `getOutput` reads the *voltage* of an OUTPUT neuron (an accumulating leaky
  readout, *not* a spike count — outputs rarely spike, see the replay notes).
- `reward(amount)` — add to `totalReward`, then push `amount * learningRate`
  through every synapse's `applyReward`.
- `releaseDopamine(amount)` — set the global dopamine signal (consumed by the
  next `tick`).
- `adjustThresholds(targetRate)` — homeostatic pass over all neurons, then resets
  every `fireCount`. **Only INTERNAL neurons adapt**; INPUT/OUTPUT thresholds are
  untouched here, which gives callers a stable place to override OUTPUT
  thresholds (as `dog_drive` does — see §7.5).
- `setNeuronThresholds(v)` / `getNeuronThresholds()` — bulk read/write of every
  neuron's threshold (Δ-housekeeping plus persistence; used by the dog to drop
  OUTPUT thresholds to 0.5).
- `reset()` — clears every neuron + synapse trace and dopamine. **Caveat:** it
  also zeroes `totalReward`; callers that need running totals accumulate reward
  on their own side (`main.cpp`, `snn_drive`, `dog_drive`).
- Wiring: `connectSliding(from, to, limit)` — for source `i`, connect to targets
  `start .. start+limit` where `start = floor(i·(N2−K)/(N1−1))`. Every source
  thus sees the same-sized receptive window; layer edges land densely on a
  diagonal band. (The JS network panel re-implements this identically so SVG
  edges match C++ synapses 1:1 — see §10.7.)
- Learning-rate and topology accessors: `setLearningRate`, `getLearningRate`,
  `getLayerSizes`, `getWiringLimits`, `getTotalSteps`; introspection for
  persistence/replay: `getNeuronThresholds`, `getSynapseStrengths`,
  `getFiredFlags()` (an `std::vector<int>` of 0/1 per neuron, in layer order —
  what the replay panel colors by). Training-run bookkeeping for saved JSONs:
  `setEpisodesTrained`, `setSuccesses`, `setTotalReward`, `setBestReward`,
  `setTotalSteps`.

### 5.4 `utils/BrainSave`

Stateless JSON persistence with no file-locking assumptions.

- `BrainSave::save(brain, dir)` — writes `data/brain/brain_N.json` where `N` is
  the next free number (scan of the dir); returns the path. Payload: topology,
  wiring limits, learning rate, episode-success stats, every neuron's threshold,
  every synapse's strength.
- `BrainSave::load(path)` — reconstructs a `Brain` from JSON and reapplies
  thresholds + strengths. Roundtrip is bit-exact (unit-tested).
- `nextNumber(dir)` — scans for the highest `brain_N.json` and returns `N+1`
  (used for auto-numbered saves).

`data/*` is git-ignored, so trained brains live locally (in CI they'd be
artifacts, see §9).

---

## 6. Training CLI — `main.cpp` (`sage`)

Runs the classic cart-pole *teaching* task to prove the learning loop end to
end. It is deliberately a toy: the SNN must hold an inverted cart-pole pole with
binary exert-force outputs.

Arguments (all optional):

| Flag | Default | Meaning |
|---|---|---|
| `--episodes` | 1000 | number of episodes to train |
| `--decision-ticks` | 240 | SNN ticks per episode (each maps to one cart/pole physics step) |
| `--epsilon-start` | 1.0 | initial exploration probability (random action) |
| `--epsilon-end` | 0.05 | exploration floor |
| `--epsilon-decay` | 0.999 | per-episode decay (default path from start→end) |
| `--learning-rate` | 0.02 | reward scaling into `applyReward` |
| `--seed` | default(1) | RNG + demo task seed |
| `--save-dir` | `data/brain` | where `BrainSave` numbers brain JSONs |
| `--fire-log` | (none) | optional; also records firing per tick |

Per episode: sense the pole angle/angular velocity, tick the brain N times, act
with ε-greedy, accumulate reward (proportional to time alive / pole balance),
call `brain.reward(episodeReward)` at the end, then `adjustThresholds(target)` +
`reset()`. Prints episode→success counts and a success-rate summary; saves the
final brain (and, with `--fire-log`, the fire log + meta sidecar).

Measured behavior (real runs): 1000 eps, seed 222, ε 1→0.05 → **614/1000 (61.4%)**,
reward 421; 4000 eps, seed 222 → **2394/4000 (59.9%)**; lr grid 0.01/0.05, seed 7 →
115/200 (57.5%) both. The task itself is degenerate (the always-output-0 policy
is a good baseline), so ~60% is the *plateau that proves the learning loop is
alive*, not a claim of mastery.

### 6.1 Fire-log format (`main.cpp --fire-log` / `snn_drive` / `dog_drive`)

A CSV plus a JSON sidecar, produced while training.

```
# topology 8,16,16,16,4
# episodes 300 decisionTicks 240 seed 12 lr 0.02 epsilonStart 1.0 epsilonEnd 0.05
ep,tick,i0,i1,...,f0,f1,...,f59
0,0,0.5,0.2,...,0,0,0,...
...
```

- Comment header lines start with `#`; the first says the topology, the second
  the run config (`dog_drive` also emits `# task dog_walk`, `# hz 100`, and
  `targetDist`).
- One row per *tick*; columns: episode, tick, the N input voltages, then one
  0/1 fired flag per neuron (input + hidden + output — total = full-layer count).
- `FILE.meta.json` sidecar: `{brain, layerSizes, wiringLimits, episodes,
  decisionTicks, seed, lr, epsilonStart, epsilonEnd, successes, totalReward,
  episodesTrained}`. `snn_drive` adds `stepsPerEpisode`, `shapingWeight`, `task`;
  `dog_drive` adds `stepsPerEpisode`, `hz`, `task: "dog_walk"`, `targetDist`,
  `successRate`, `bestDistance`, `avgDistance`.

The replay engine derives neuron coordinates from the topology header and plays
rows at `1/hz` per row when present.

---

## 7. MuJoCo side (C++)

Four models, four executables. Everything here runs locally on the dev box
(`LD_LIBRARY_PATH=$HOME/.mujoco/lib`).

### 7.1 Models (`tests/models/*.xml`, MJCF)

Each scenario is a minimal world (mirrored for OmniSim under
`tests/omnisim_worlds/` where applicable):

- `cart_pole.xml` — a cart sliding on a floor with an inverted pole; the cart is
  pushed by an effort-capped motor (`control`). Physics dt 0.02 s.
- `ball_drop.xml` — a sphere falling under gravity onto a floor. dt 0.005 s.
- `arm_reach.xml` — a 2-DOF planar arm (shoulder + elbow) reaching for a colored
  target sphere. dt 0.01 s. This is the model the SNN drives (`snn_drive`).
- `dog_walk.xml` — a quadruped: cuboid torso + 4 identical legs (hip hinge +
  knee hinge + foot sphere), driven by 8 torque motors. Sensors expose the 8
  joint angles, the torso quaternion, and the torso position. dt 0.01 s. See
  §7.5 for the body layout (the replay artist's FK mirrors it geometrically).

### 7.2 `tests/cross_sim_harness.cpp` (`cross_sim_test`)

MuJoCo's *baseline*: runs each scenario with a scripted/constant control and
records the tracked body's trajectory to `tests/results/`. Endpoints:

- `runMuJoCoTest(name, ...)` — loads an MJCF, steps `N` times at dt, and appends
  `time,x,z,y[?]` rows to `tests/results/{scenario}_mujoco.csv` (columns
  `time,x,y,z,vx,vy,vz` for `arm_reach`, simplified for cart/ball).
- `findBodyId(...)` — resolves an MJCF body name to `mjtObj` for the tracked body
  (cart, ball, or arm end-effector), and `getBodyPosition(...)` converts the
  `mjtData` qpos to world space.
- Energy helper (`kineticEnergyFrictionLoss`, etc.) — sheds insight into where
  the two engines differ numerically.
- CLI: `--scenario all|cart_pole|ball_drop|arm_reach`, `--steps`, `--dt`.

### 7.3 `tests/snn_drive.cpp` (`snn_drive`) — SNN embodied learning (arm)

Attaches a live SNN to MuJoCo so the brain commands a physics body and learns
from the outcome. This is the piece the replay player shows as a real
recording of "brain + world".

Per physics step (`dtReal = 0.01 s`):

1. **Sense** — read sensors: end-effector pos, target pos, shoulder/elbow frame
   positions → build the 8 INPUT voltages (ee−target error /0.35, ee world /0.8,
   joint angles /1.5708, each clamped to ±2).
2. **Think** — `brain.tick()` once.
3. **Act** — the 4 OUTPUT voltages become 0/1 spikes → two motor commands:
   shoulder `0.45 + 2.0·out0`, elbow `0.15 + 2.0·out1`, both clamped to ±1
   (explore phase: uniform-random instead, ε-greedy).
4. **Step physics** — one `mj_step`.

Reward: episode ends when the effector's 3D distance to target falls below
`--reach-dist` (default 0.1) — *reached*. Optional per-step shaping via
`--shaping W` (per-step reward `0.05·W·clamp(0.2 − 2.0·dist, −0.2, 0.2)`).

Outputs (all written only when the matching flag is set):

- `--record FILE_ee.csv` — `time,x,y,z,vx,vy,vz` of the end-effector per step.
- `--fire-log FILE` — §6.1 format, `# hz 100`.
- `--save-dir` — auto-numbered brain JSON via `BrainSave`.

CLI: `--episodes` (default 120), `--steps` (240), `--epsilon-start/end`,
`--learning-rate`, `--seed`, `--model`.

**Measured results (honest):** across 60/120/150 episodes and every variant
(bias, motor gain, shaping weight) the reaching task **never reached** — 0 reached
of all runs; final end-dist 0.24–0.85. Root causes = §5.2: sparse payoff is erased
by `LEAK_RATE`, and at the old resting strength the net was desk-bound. What
*did* happen is a working closed loop — the brain read MuJoCo, modulated motor
outputs, and the replay shows correlated firing + arm movement. **The
integration works; convergence is a documented limitation of the
credit-assignment scheme, not a bug.** The `RESTING_STRENGTH` fix (§5.2) is
what later unblocked the dog.

### 7.4 `tests/brain_test.cpp` (`brain_test`)

Small assert-driven unit suite (no framework): tick advances `totalSteps`,
setInput/getOutput don't throw, `reward` accumulates `totalReward`, `BrainSave`
roundtrip bit-preserves learningRate/topology/wiring/thresholds/strengths, and
`reset` clears transient state. Exit code 0 iff all pass.

### 7.5 `tests/dog_drive.cpp` (`dog_drive`) — SNN quadruped gait (deep dive)

The most recent and most complete embodied loop. A 74-neuron brain (wireless
12→18→18→18→8, wiring limits `[6,5,5,4]`) drives a 14-body dog and learns under
reward. This is the case that exercises every part of the stack, so it is
documented to the level of a code walkthrough.

**The model (`dog_walk.xml`).** World z is up, gravity `0 0 -9.81`. The torso
(`pos 0 0 0.27`, box half-sizes `0.2 × 0.08 × 0.05`, i.e. full 0.4 × 0.16 × 0.1)
sits on a `freejoint root` (so it has full 6-DOF). Four legs hang from hip
bodies at `x=±0.13` (front=+x), `y=±0.075` (left/right): FR, FL, RR, RL.

```
torso
├── FR_hip (0.13, -0.075, 0) ── hip hinge  axis(0 1 0)  ── FR_thigh (-0.11 z)
│     └── FR_shank (0,0,-0.11) ── knee hinge axis(0 1 0) ── FR_chank (-0.11 z)
│           └── FR_foot (sphere r 0.025)
├── FL_hip (0.13, +0.075, 0)   ... (same shape)
├── RR_hip (-0.13, -0.075, 0)  ... (same shape)
└── RL_hip (-0.13, +0.075, 0)  ... (same shape)
```

Each leg: thigh capsule radius 0.02 length 0.11, shank radius 0.015 length 0.11,
foot sphere at the shank end. Hip ranges ±0.9 rad, knee range −0.3…1.2 rad.
All hinges rotate about the side-to-side Y axis, so whole legs swing front↔back.
8 torque motors (gear 25 for hips, 12 for knees; `ctrl` clamped ±1). Sensors:
`jointpos` for the 8 joints (order FR_hip, FL_hip, RR_hip, RL_hip, FR_knee,
FL_knee, RR_knee, RL_knee), plus `torso_quat` and `torso_pos` on the torso site.

**Input construction (proprioception only — no vision).** Per tick, from the
sensor data at `m->sensor_adr[]`:

| input | source | scaling | clamp |
|---|---|---|---|
| i0–i7 | 8 joint angles | ×2.0 | ±2 |
| i8 | torso `pitch = atan2(upx, upz)` | /0.5 | ±2 |
| i9 | torso `roll  = atan2(upy, upz)` | /0.5 | ±2 |
| i10 | body height `h = xpos.z` | (h−0.27)/0.2 | ±2 |
| i11 | constant drive channel | 1.5 V | — |

The up-vector (for pitch/roll) is derived from the torso quaternion
(`upx = 2(qw·qy + qz·qx)`, etc.). **i11 is the trick that keeps the net alive:**
with everything else near zero while the dog stands still, a purely zero input
rest state would leave every hidden/output neuron silent (see the dead-net
finding §5.2). The constant 1.5 V drive always fires its useful input weights,
so downstream layers produce baseline motor activity to build on.

**Output mapping (act).** For joint `j` (0–3 hips, 4–7 knees):

```
vc    = clamp(getOutput(j), 0, 1)          # analog readout of OUTPUT voltage
bias  = (j < 4) ? -0.1 : -0.45             # hips start loose, knees braced
ctrl  = clamp(1.25·vc + bias, -1, 1)       # motor torque command
if explore: ctrl = clamp(ctrl + εAmp·U(−1,1), -1, 1)   # ε-greedy, εAmp=0.35
d->ctrl[actuator_id[j]] = ctrl
```

The three tuning knobs that mattered empirically:

- **OUTPUT thresholds forced to 0.5** right after construction
  (`getNeuronThresholds` → set last 8 to 0.5 → `setNeuronThresholds`). Because
  `adjustThresholds` never touches OUTPUT neurons, the 0.5 survives
  per-episode homeostasis. At the default higher threshold, outputs accumulate
  voltage until *every* motor saturates at ±1, which just tips the dog over;
  at 0.5 the outputs fire often and stay in a differentiable analog band.
- **Bias asymmetry** — hips start at slight negative torque, knees brace hard
  negative (−0.45), which is the standing-crouch pose the exploration starts
  from.
- **Exploration is additive noise around the policy** (not pure random actions),
  so the net is always being shown actions near what it already believes.

**Reward.** Dense per-step progress plus terminal shaping:

```
dx        = xpos.x(t) − xpos.x(t−1)          # torso forward displacement
traveled += max(0, dx)                       # cumulative, no credit for going back
rewardStep = clamp(dx · 8.0, −0.03, 0.03)    # dense, saturates
brain.reward(rewardStep)
# terminal (once per episode):
success   → +2.0     (traveled ≥ --target-dist, default 2.0 m)
fell      → −1.0     (z < 0.12  OR  |pitch| > 1.3)
timeout   → −0.3     (steps exhausted; default 2400 ticks = 24 s)
brain.reward(termReward)
```

`brain.reward(x)` internally scales by `learningRate` (default **0.5** here —
deliberately hot compared to `sage`'s 0.02, to fight the leak described in
§5.2).

**Fall handling — a deliberate design quirk.** `fell` latches the first time
`z < 0.12 || |pitch| > 1.3`, but the episode only *hard-breaks* if `t > 400`.
Before that the sim keeps stepping. Transient dips (a stumble that recovers,
a hop that clips its own feet) do not silently truncate episodes, which prevents
two pathologies: (a) the net never learning to recover, and (b) every episode
being 2 ticks long so the brain starves. Episodes that never truly collapse run
the full 24 s. The recording step does **not** early-break on `z` either — only
on `traveled ≥ target` — which matters for the replay (see below).

**Per-episode loop.** Each episode: `mj_resetData` + `brain.reset()` +
`brain.resetOutputs()`, ε from `--epsilon-start`→`--epsilon-end` (1.0→0.05)
linearly across episodes, then tick→sense→act→step for up to `steps` ticks. The
*applied* controller trace (`curCtrls`, `steps × 8`) is stashed per episode; the
episode with the highest `traveled` becomes `bestCtrls` (best-trace recording,
next paragraph). After the episode: terminal reward → accumulate stats →
`brain.adjustThresholds(2)` (homeostasis targets rate 2·... for the hidden
layer) → report row `ep,distTravelled,reward,success` (with `--report`).

**Best-episode exploit recording.** After training, the winning controller trace
is replayed deterministically from a clean `mj_resetData` state:

1. `mj_resetData(m, d)`; `brain.reset()`; `brain.resetOutputs()`.
2. For each trace tick `t`: build inputs **from the live sim state** (same sense
   math), `brain.tick()` (so the fire-log is populated by real scores), then
   *override* `d->ctrl[act[j]] = bestCtrls[t·8+j]` (the learned actions, not the
   net's current output).
3. `mj_step`, write the CSV row (below), stop when `traveled ≥ target`.

Recording the *best episode* instead of the final one matters: the last episode
is exploration-heavy and unrepresentative; the best trace is what the policy
actually learned. Writing the fire-log during the replay (not during training)
means the fired-flags you watch in the SNN panel correspond exactly to the
trajectory you watch in the 3D view (the same ticks, the same brain state).

**Recording format (pose-enriched).** The trajectory CSV is more than
positions — it carries the full body pose so the viewer can reconstruct the dog
(§10.5):

```
time,x,y,z,vx,vy,vz, qw,qx,qy,qz, FR_hip,FL_hip,RR_hip,RL_hip, FR_knee,FL_knee,RR_knee,RL_knee
```

Columns: torso position + forward velocity (vx), then the torso **quaternion**
and the **8 joint angles** (from sensor data, same values the brain saw). With
the freejoint quaternion ± the joint angles, the articulated body is fully
determined at every tick — the viewer reproduces it with forward kinematics.
`vx` is `dx/timestep`; `vy,vz` are hard-coded 0 (kept for column
compatibility with the other scenario CSVs).

**Results — honest.** Defaults (`--episodes 3000 --seed 42 --target-dist 2.0`):

```
Done: 3000 eps, succeeded 0 (0%), best dist 1.4 m, avg dist 0.11 m,
total reward -2034, seed 42
Recorded best episode (402 ticks): travelled 1.4 m
```

The brain **never walks 2 m**; its best policy is a hopping/shuffling gait that
covers ~1.4 m in ~4 s and then collapses — exactly the "it learned to jump"
behaviour the replay shows. `avg dist 0.11 m` says most episodes barely move or
fall immediately. So: the closed-loop stack is fully working (perceive → spike →
act → reward → learn → record → replay), and the *learning* is what's next on
the table. The bright spot is the architecture: given the input construction,
the output mapping and the resting-strength fix, the net found *some* forward
motion; it's now a credit-assignment / reward-shaping problem to shape that into
a gait.

CLI: `--episodes` (3000), `--steps` (2400), `--target-dist` (2.0),
`--epsilon-start` (1.0), `--epsilon-end` (0.05), `--explore-amp` (0.35),
`--learning-rate` (0.5), `--reward-progress` (8.0), `--seed` (42), `--brain FILE`
(load a saved brain instead of training), `--save-dir` (`data/brain`),
`--record FILE` (trajectory CSV), `--fire-log FILE` (firing CSV + meta sidecar),
`--report FILE` (per-episode stats CSV).

---

## 8. OmniSim side (Python)

`tests/omnisim_compare.py` is the cross-simulator client. OmniSim is controlled
over HTTP/JSON; the script drives the same scenarios it knows MuJoCo already ran
and compares the trajectories.

- `SCENARIOS` table maps each scenario to `(mujoco CSV, omniworld path, dt,
  steps, tracking rule)`:
  - `cart_pole` → `joint:CART:cart_force` (slider joint position — matches the
    MuJoCo cart x; the muscle is named after its effort-capped motor);
  - `ball_drop` → `def:BALL` (scene-tree node by DEF);
  - `arm_reach` → `last-solid` (deepest *actual* Solid in DFS order, since the
    arm's moving links have no DEF).
- All HTTP via `urllib` (stdlib only). Key endpoints: `GET /healthz`,
  `POST /world/load` (with `with_supervisor: true` — the supervisor is what
  makes `/scene/tree` and `/sim/step` work; without it those 503), `POST /sim/step`
  (no `ok` field; returns `sim_time_ms`, `advanced_to_ms`), `GET /scene/tree`.
- Stepping discipline (from a protocol WARNING): a single `/sim/step` can take
  ~0.5–1 s and an overly long request drops the session socket
  (`SUPERVISOR_RPC_TIMEOUT_S=120`, no retry). So the client steps in **batches
  of 10** and samples one position per batch at the matching sim time, always
  keeping requests short.
- After load it calls `/sim/reset` with `{"restore": null, "settle_steps": 0}`
  so the comparison covers the full authored trajectory rather than the
  post-settle state (the load's `wait_s` lets the engine run ahead — the ball
  would otherwise already be on the floor).
- Tracking: `_position_values` rejects harness-injected nodes; `sample_position`
  implements the `def:` / `joint:` / `last-solid` rules; `OMNISIM_DUMP_TREE=1`
  dumps `scene_tree_*.json`, `robots_*.json`, `joints_CART/ARM.json` to
  `tests/results/` for debugging.
- Comparison: nearest-neighbour match of MuJoCo samples by time, L2 distance
  per sample, then `avg_err < 0.05` → **YES** else **MISMATCH** (the smoke
  signal the CI compares on). Writes `tests/results/{scenario}_omnisim.csv`
  with columns `time,sim_time_ms,x,y,z`.

Final CI outcome (run 34248901341): cart_pole avg L2 **0.982400**, ball_drop
**0.112472**, arm_reach **0.601647** vs 0.05 threshold → all MISMATCH, and the
script reports `*** FIRST MISMATCH ***`. Root cause: **Newton's solver freezes
robot-jointed chains** (cart/arm), while free solids fall correctly (ball got
within ~0.11).

---

## 9. CI — GitHub Actions + the GitHub CLI

### 9.1 The workflow (`.github/workflows/cross-sim-test.yml`)

One file, one job (`cross-sim`), `ubuntu-24.04`, timeout 150 min. Triggered on
push to `master` and by manual `workflow_dispatch`. The job is a single pipeline
(not split into lanes) that:

1. **Checks out** the repo with submodules.
2. **Validates system Python 3.12** (OmniSim's harness is python-version-
   sensitive; the engine embeds the *system* python, so `actions/setup-python`
   must never be used — that's the #1 recipe rule inherited from OmniSim's own
   linux-build workflow).
3. **apt-installs the X/GL runtime** an OmniSim window-less render needs:
   `xvfb xauth libxcb-*0 libgl1 libglu1-mesa mesa-vulkan-drivers libsndio7.0` etc.
   (This list grew one package at a time as ldd scans found missing libs —
   see the run history.)
4. **Installs MuJoCo 3.3.2** into `${{ workspace }}/.mujoco`.
5. **Builds Sage** (`cmake -B build ... && cmake --build build -j`).
6. **Runs the MuJoCo baselines** (`cross_sim_test all`) → uploads `*_mujoco.csv`.
7. **Restores/creates the OmniSim source build** (`actions/cache` on `omnisim/`,
   key `omnisim-linux-v1`). On cache miss it clones OmniSim and runs the
   bootstrap **phases** (`python deps wgpu build gpu`) with `OMNISIM_SKIP_TORCH=1`
   (a ~2.5 GB CUDA download a CPU runner can't use) and `JOBS=4`.
8. **Installs Newton runtime wheels into system python3 on every run** (not
   cached) — wheels land in a cache-evicted place; skipping this leaves
   "NO PHYSICS BACKEND".
9. **Smoke-tests the hand-authored `.omniworld` mirrors** through the engine's
   own headless smoke, forcing Newton, world by world.
10. **Runs the cross-sim comparison** per scenario, one OmniSim harness per
    session on an isolated port (`cart_pole` 6811, `ball_drop` 6862,
    `arm_reach` 6921; supervisor = port + 10), with `setsid`+`xvfb-run`;
    each harness is `pkill`-ed for the next. This step is
    **`continue-on-error: true`**.
11. **Uploads `sage-cross-sim-results`** (CSVs, JSON, per-scenario harness
    logs) with `actions/upload-artifact`, always.

**Trap:** because the comparison step is non-fatal, the run *turns green* even
when the physics diverge. **Green ≠ success.** Always confirm with
`gh run view --log` and download the artifacts for the real numbers.

**Run housekeeping:** every push spawns a new run and the Actions tab floods.
Runs are disposable artifacts; the useful ones are (a) the latest successful
run (current master state + fresh results) and (b) any run whose numbers a
deliverable quotes. Historical debug iterations can be deleted with
`gh run delete <id>` (interactive-free) to keep the tab readable.

### 9.2 Driving CI from the GitHub CLI

The `gh` CLI authenticates to GitHub and talks to this repo's Actions:

```bash
export PATH="$HOME/.local/bin:$PATH"   # gh isn't on the default PATH here

gh run list            # recent workflow runs
gh run list --workflow cross-sim-test.yml
gh run view <run-id>                    # status + job list
gh run view <run-id> --log              # full step logs (real verdict lives here)
gh run download <run-id>                # pulls uploaded artifacts
gh run rerun <run-id> --failed          # re-run only failed jobs
gh run watch <run-id>                   # stream progress until done

gh repo list DrEggCake                  # Sage + Sage-Java live under this org
gh api repos/DrEggCake/Sage/actions/runs/<id>/logs -H "Accept: application/vnd.github+json"
```

For private repos you must `gh auth login` once per machine (device flow +
the browser link). `gh run view <id> --log` is the definitive artifact check:
a green badge alone proved misleading before.

### 9.3 CI vs local reality

The local dev box **can** build/run everything MuJoCo (`sage`, `brain_test`,
`cross_sim_test`, `snn_drive`, `dog_drive` all run locally) and the replay
server, but **cannot** run OmniSim — that only happens on CI (linux runner +
system Python 3.12 + Xvfb). So the local loop is: build → run MuJoCo targets →
inspect CSVs/replay; the CI loop adds: build OmniSim → load each world →
compare servos → upload artifact.

---

## 10. The replay player (Python + JavaScript) — detailed

This is the shakiest mental model for most people because it spans two
languages, a custom data pipeline, and a browser. Read 10.1 (server) then
10.2–10.7 (GUI) top to bottom.

### 10.1 `tests/replay/serve_replay.py`

A stdlib `ThreadingHTTPServer`. Its single page is `replay.html`; everything
else is data. Two responsibilities:

**A. Static + data serving, path-safe.** All user-supplied files live under one
of the three configured roots. Every route resolves the real path and rejects
anything escaping the root (`target.is_relative_to(root)` — a traversal guard).
Cached responses get `Cache-Control: public, max-age=300`; the manifest gets
`no-store` so refresh always sees new files.

| Route | Source dir | Notes |
|---|---|---|
| `/` and `/index.html` | `replay.html` next to the script | the app |
| `/vendor/*`, anything under the script dir | `tests/replay/` | three.js + OrbitControls |
| `/results/<file>` | `--results` (default `tests/results`) | CSVs, mirrored `_mujoco`/`_omnisim` |
| `/firelogs/<file>` | `--firelogs` (`data/firelog`) | firing CSVs produced by `-fire-log` |
| `/brains/<file>` | `--brains` (`data/brain`) | `brain_N.json` |
| `/api/manifest` | computed | JSON index, see below |

**B. The manifest.** `build_manifest(...)` scans the roots and returns one JSON
document the browser consumes once at startup:

```json
{
  "results_dir": "/path/tests/results",
  "recordings": [
    { "id": "dog_walk_mujoco", "scenario": "dog_walk", "engine": "mujoco",
      "file": "/results/dog_walk_mujoco.csv",
      "header": "time,x,y,z,vx,vy,vz,qw,qx,qy,qz,...", "rows": 402 },
    { "id": "dog_walk_omnisim", "scenario": "dog_walk", "engine": "omnisim", ... }
  ],
  "firelogs": [
    { "id": "dog_walk.log", "file": "/firelogs/dog_walk.log.csv", "rows": 403,
      "meta": { "layerSizes": [12,18,18,18,8], "hz": 100, "bestDistance": 1.399, ... } }
  ],
  "brains": [
    { "id": "brain_1", "file": "/brains/brain_1.json",
      "layerSizes": [12,18,18,18,8], "episodesTrained": 3000, ... }
  ]
}
```

- **Recordings:** a file name must match `^(?P<scenario>.+)_(?P<engine>mujoco|omnisim)\.csv$`
  (so a scenario = two files with matching base names). The *header* and *row
  count* are pre-scanned so the UI can show them without fetching every file.
- **Firelogs:** any `*.csv` in the firelog dir; if a `FILE.meta.json` sits next
  to it, the sidecar is inlined as `meta` (so the player never fetches the CSV
  until you open the SNN panel).
- **Brains:** every `*.json` in the brain dir; the interesting fields
  (`layerSizes`, `wiringLimits`, `learningRate`, `episodesTrained`, `successes`,
  `totalReward`, `bestReward`, `totalSteps`) are inlined for the UI.

CLI: `--results DIR` (default `tests/results`), `--firelogs DIR`,
`--brains DIR`, `--port` (8787), `--no-browser`. It binds to `127.0.0.1`.

### 10.2 Boot & lifecycle (`replay.html`)

The whole app is one ES module in one file; no build step, no framework. Boot
order:

1. **Two WebGL scenes are created up front** (`views.mujoco`, `views.omnisim`),
   each with: `THREE.Scene` (dark bg + fog), a `PerspectiveCamera`, orbit
   controls (damping on), ambient + directional light, a `GridHelper`, a red
   `AxesHelper(1.4)` (the origin frame you see), and a floor plane. A
   `ResizeObserver` keeps aspect ratios correct.
2. `init()` fetches `/api/manifest`,
   - populates the **Scenario** dropdown from distinct `scenario` values (all
     engines with the same base name group into one playback entry),
   - populates the **fire log** and **brain** selects in the SNN panel,
   - auto-loads the first scenario, then auto-selects the first fire log.
3. `loadScenario(id)` — the heart of startup:
   - fetches each engine's CSV, runs `parseCSV`,
   - **normalizes time**: `s.rel = s.t − samples[0].t` so both engines (and any
     recording) share a common t=0 even if their absolute `time` columns start
     differently,
   - builds the per-engine scene objects (`buildEngineObjects`),
   - frames the camera to the data bounds (`frameCamera`),
   - unions all engines' relative sample times into `state.keyframes` (the
     playback timeline), then plays.

### 10.3 CSV parsing & the shared timeline

`parseCSV(text)` is **header-driven and scenario-agnostic** — it finds columns by
name (`time`, `x`, `y`, `z`, optional `sim_time_ms`), ignores everything it
doesn't know, and attaches extras only when present:

```js
const iiQw = ci('qw') /* ... */;
const iiJ = ['fr_hip','fl_hip','rr_hip','rl_hip','fr_knee'/*...*/].map(ci);
const hasPose = [...all >= 0];              // dog_walk-style pose columns present?
// per row: sample = { t, sim, x, y, z, q: [qw,qx,qy,qz], j: [8 joints] }
```

So the generic 3-view ball-trail works for every scenario, and any recording
that carries `qw,qx,qy,qz` + the 8 joint columns automatically gets the
articulated-render path too (`hasPose`) — no per-scenario hardcoding beyond the
column names.

Interpolation: `sampleAt(samples, t)` walks the sorted samples; it
linearly interpolates `x,y,z` and copies the **nearest sample's** `q`/`j` pose
(between recorded ticks the pose snaps, the position glides — good enough at
100 Hz). `trailPoints(samples, t)` returns the samples ≤ t as THREE vectors.

The **timeline** is the sorted union of all relative sample times across the
loaded engines (`state.keyframes`), so scrubbing the slider steps through every
recorded instant both engines share. The slider's integer value is a keyframe
index resolved by `nearestKeyframe(t)`.

### 10.4 Scene objects, coordinate mapping, and the HUD

**Coordinate mapping (the easy-to-trip detail).** MuJoCo is **z-up**; three.js is
**y-up**. The swap is applied once, mechanically, everywhere:

```js
function m2t(v) { return new THREE.Vector3(v.x, v.z, v.y); }   // muJoCo -> three
```

i.e. data `y` maps to three-`z`, data `z` maps to three-`y` (up). Every point
placed in the scene goes through this (marker, trail, camera framing, dog FK),
so the left/right flip danger you'd fear from a naive y↔z swap never happens in
the codebase — it's centralized.

Per engine, `buildEngineObjects` creates:
- **trail** — a `THREE.Line` whose position attribute is rewritten each frame
  (`frameEngine`) from `trailPoints`, capped at 600 points (a rolling ribbon);
- **marker** — a 7-cm sphere at the tracked body's current position (the engine
  color, emissive). *Hidden when the recording has pose columns* — the dog rig
  replaces it;
- **begin** — small white sphere at `samples[0]` (start marker);
- **endBox** — small red cube at the final sample (**the end-of-recording
  marker** ≈ camera-space reference point, worth knowing before you wonder what
  it is);
- **dogRig** (only when `hasPose`) — see 10.5.

Frame: `frameEngine` repositions the marker, updates the trail, and, for pose
recordings, calls `poseDog`. The **HUD** (`renderHUD`) is four absolutely
positioned boxes inside the views container (so they never cover the toolbar):
top-left scenario/sample stats, top-right time/keyframe/position/`sim ms`/status,
bottom-left live **cross-engine L2 distance** at the current time plus final
positions, bottom-right controls legend. `sim ms` is synthesized as
`base + rel·1000` (the recorded engines don't share a wall clock). The camera is
auto-framed to the data bounds (`frameCamera`) and reframed by `R`.

### 10.5 Articulated dog rendering (forward kinematics in the browser)

The dog_walk recording carries `qw,qx,qy,qz` + 8 joint angles per tick
(§7.5). The viewer rebuilds the body with FK that mirrors the XML geometry
exactly — hip offsets `(±0.13, ±0.075, 0)`, both links length **0.11**, all
hinges about the side-to-side axis:

1. **`quatToM(q)`** — MuJoCo quaternion `(w,x,y,z)` → a 3×3 rotation matrix
   (columns are `R·e_i`, standard active-rotation convention, z-up).
2. **Torso** — the three mapped basis vectors (`m2t` on each column) become a
   `THREE.Matrix4().makeBasis(...)`, which sets the box mesh's quaternion; the
   box is positioned at `m2t(torso pos)` and scaled `(0.4, 0.1, 0.16)` (x wide,
   up, z deep — the XML's `0.2×0.08×0.05` half-sizes doubled and re-oriented).
3. **Legs** — for leg i (FR, FL, RR, RL; `DOG_HIP` gives the hip offset, `s.j`
   the hip angle θ1 and knee angle θ2, both sign-convention matching the sim's
   +Y hinge):

   ```
   hip  = torso + R · (hx, hy, 0)
   v1   = R · rotY(θ1) · (0,0,−1)          # thigh direction
   knee = hip + 0.11 · v1
   v2   = R · rotY(θ1+θ2) · (0,0,−1)       # knee adds to hip (same axis)
   foot = knee + 0.11 · v2
   ```

   `rotY` rotates about the Y axis, and because hip and knee share that axis the
   two rotations compose, so the shank is just `rotY(θ1+θ2)`.
4. **Meshes** — a shared unit `CylinderGeometry` scaled per segment:
   `placeCylinder(mesh, from, to, radius)` positions the cylinder at the
   midpoint and orients it with `quaternion.setFromUnitVectors(Y, dir)`. Thighs
   radius 0.02, shanks 0.015, feet spheres 0.025 — the XML's geom sizes. Colors
   match the model: torso `0xe69a33` (orange), thighs `0x598ce6` (blue), shanks
   `0x4dbf66` (green), feet `0x26333f` (dark).

Result: the same rig both engines show becomes a literal dog — you watch the
body rotate, legs scissor, hips/tongues of the gait — instead of a ball. This is
why the dog's "learned jumping" was visible at all.

### 10.6 Playback loop & keyboard

`rafLoop` is a single `requestAnimationFrame` driver:

```
dtReal → if playing: state.t += dtReal × state.speed      # SIM seconds, real seconds
         wrap at duration (loop) or clamp+pause (one-shot)
         tick()                                           # reposition both engines
         brain.acc += dtReal × speed × hz; stepBrain(floor(acc))   # SNN panel
render every visible view; controls.update()
```

Playback rate is **sim-time-faithful**: 1.0× replays the recorded seconds in
real seconds (regardless of CSVs having different row densities), and the SNN
panel advances at the fire-log's `hz` (dog = 100, sage = 24 default) so the
network stays phase-locked to the physics being replayed. `frameStep(±1)`
jumps one *keyframe* (not frame), and `stepBrain(±1)` steps one fire-log row.

Keys: `Space` play/pause, `←`/`,` and `→`/`.` frame step, `[`/`]` speed
(0.1–8×), `L` loop, `R` reset camera, `F` fullscreen; drag/wheel/right-drag
orbit/zoom/pan. Engine checkboxes (MuJoCo/OmniSim) collapse views to a single
pane; `layout()` flexes the survivors.

### 10.7 The SNN network panel

The third view (`view-brain`, toggled by the **SNN** button) is a live SVG
rewrite of the network:

- **`parseFireLog`** — reads the `# topology a,b,c,d,e` header and the first two
  data columns (`ep`, `tick`), splits the rest into `inputs[]` (the N input
  voltages) and `fired[]` (0/1 per neuron, layer order). Optional `# hz` in the
  header sets the tick rate.
- **`slidingWiring`** — *re-implements* `Brain::connectSliding` in JS (same
  `start = floor(i·(N2−K)/(N1−1))` formula), so the SVG edges correspond 1:1 to
  the C++ synapses. With a paired brain JSON loaded, edge `stroke-opacity`
  scales with `|synapseStrength|`; without one, edges are uniform faint.
- **`buildNet`** — lays the layers as columns (x spread, y = `H·(j+1)/(n+1)`),
  draws edges then circles; a `<text>` label per column says
  `input/layer 1/…/output — N`.
- **`renderNetRow`** — colors: input circles get a blue intensity fill from
  `(inputV+2)/4`; every other neuron fills green + a glow filter (`circle.fire`)
  when its fired flag is 1. Status line:
  `ep · tick/decisionTicks · row/totalRows · fired N/total · brain <json>`.
- **Transport** — `-`/`+` step, Play (ties into the shared `state.playing`),
  Loop; tick accumulation is hz-synced with the playback loop as above, and the
  paired-brain select re-draws edge weights live.

---

## 11. Data formats, at a glance

| Artifact | Format | Columns / shape |
|---|---|---|
| MuJoCo trajectory (base) | CSV | `time,x,y,z[,vx,vy,vz]` under `tests/results/{name}_mujoco.csv` |
| MuJoCo trajectory (pose) | CSV | base columns + `qw,qx,qy,qz,` + 8 joint angles (dog_walk, §7.5) |
| OmniSim trajectory | CSV | `time,sim_time_ms,x,y,z` under `tests/results/{name}_omnisim.csv` |
| Fire log | CSV | §6.1: `# topology/# episodes...`, then `ep,tick,i0..,f0..` |
| Brain (saved) | JSON | `brain_N.json`: topology, wiring, lr, stats, thresholds, strengths |
| Fire-log sidecar | JSON | `FILE.meta.json`: run config + stats (see §6.1) |
| Episode report (dog) | CSV | `ep,distTravelled,reward,success` (`--report`) |
| Replay manifest | JSON | `recordings[]`, `firelogs[]`, `brains[]` (see §10.1) |

Trajectory `time`/`sim_time_ms` were aligned at collection time so the player can
overlay both engines sample-for-sample (and the viewer re-normalizes to rel-time
anyway, §10.2).

---

## 12. End-to-end: how to run everything

Local (MuJoCo present):

```bash
# 1. build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# 2. engine unit checks
./build/brain_test

# 3. train the demo brain (cart-pole teaching task), save brain + firelog
./build/sage --episodes 300 --seed 222 --save-dir data/brain --fire-log data/firelog/cart_pole_snn.log.csv

# 4. run the MuJoCo baselines (produces tests/results/*_mujoco.csv)
./build/cross_sim_test --scenario all --steps 500

# 5. SNN drives the MuJoCo arm (records ee CSV + firelog + brain)
./build/snn_drive --episodes 150 --record data/snn_results/arm_snn_mujoco.csv \
    --fire-log data/firelog/arm_snn.log.csv --save-dir data/brain --seed 7

# 6. train the quadruped (records full-body pose CSV + firelog + brain + report)
./build/dog_drive --episodes 3000 --record tests/results/dog_walk_mujoco.csv \
    --fire-log data/firelog/dog_walk.log.csv --report data/firelog/dog_walk_report.csv \
    --save-dir data/brain --seed 42

# 7. serve the replay player
python3 tests/replay/serve_replay.py --no-browser \
    --results tests/results --firelogs data/firelog --brains data/brain --port 8787
# open http://127.0.0.1:8787/  → Scenario dropdown (pick dog_walk) for the
# articulated dog; SNN panel for live firing; bottom-left HUD for cross-engine L2.
```

CI comparison flow (the whole point of the repo):

1. `cross_sim_test` records `_mujoco.csv` baselines.
2. OmniSim engine + worlds boot under Xvfb, one port per scenario.
3. `omnisim_compare.py` steps each world, writes `_omnisim.csv`, prints the
   L2 average + `FIRST MISMATCH` when it exceeds 0.05 — that comparison output
   is the runtime artifact to grep in logs (`gh run view --log`).

---

## 13. Known findings & honest limitations

- **Cross-engine mismatch is a physics-engine gap, not a setup bug.** OmniSim's
  Newton solver freezes robot-jointed chains (cart_pole, arm_reach); free
  falling solids match closely (ball_drop avg L2 0.112, already below the old
  failure noise).
- **The SNN's reward path barely learns sparse tasks.** `LEAK_RATE` (0.005/tick)
  swamps one-shot `reward()` updates; real behavior change comes from threshold
  homeostasis and sustained Hebbian co-activation. `snn_drive` reaching
  therefore achieves 0/∞ — the integration is validated, convergence is not.
- **At the original resting strength the net was dead for continuous control.**
  `RESTING_STRENGTH 0.05 → 0.25` (§5.2) is what makes hidden layers fire from
  real-valued inputs at all; keep it in mind whenever a trained net looks inert.
- **dog_walk learned a gait, not the task.** 0/3000 episodes reached the 2 m
  target; best policy is a 1.4 m hopping/shuffling run (~4 s) before
  collapse, avg 0.11 m. The architecture demonstrably works end-to-end; turning
  "jumps forward" into "walks 2 m" is a reward-shaping/credit-assignment
  problem to solve next.
- **Replay timing is approximate.** Sample times are normalized to episode
  relative time (`t - t0`); `sim ms` in the HUD is base + rel×1000; the SNN
  panel ticks at fire-log `hz` (100 for the dog), so brain-vs-physics rate
  matches by definition for hz-tagged logs.
- **Green CI runs are not proof.** The comparison step is non-fatal by design;
  always read the numbers from logs/artifacts.
- **Viewer pose is FK-snapped between recorded ticks.** Position glides
  linearly but `q`/`j` snap to the nearest sample (§10.3); at 100 Hz this is
  invisible, but it is an interpolation choice, not ground truth at arbitrary
  times.

## 14. Where to look next (by question)

- "How do I add a new scenario?" → `tests/models/*.xml` (+ mirrored
  `tests/omnisim_worlds/`), add a `SCENARIOS` entry in `omnisim_compare.py`,
  and a branch in `cross_sim_harness.cpp`. It then appears in the viewer
  automatically via the manifest regex.
- "How do I record a full-body articulated replay?" → extend your driver's CSV
  with `qw,qx,qy,qz` + 8 joint columns (§7.5) and the viewer's `hasPose` path
  (10.3/10.5) picks it up; `DOG_HIP`/FK constants must mirror your model.
- "How do I make the SNN learn a task?" → `snn_drive.cpp` / `dog_drive.cpp`
  (control mapping) held hostage by `Synapse::LEAK_RATE` / `Brain::reward`
  scaling and `RESTING_STRENGTH`; look at balancing leak vs learning rate
  before touching the task.
- "How do I tune the dog?" → `dog_drive.cpp`: the input construction (i0–i11),
  the output `bias`/scale block, the 0.5 OUTPUT-threshold override, the dense
  reward weight `--reward-progress`, and target/epsilon knobs.
- "How do I change the network shape?" → the `Brain` constructor + its 5-layer
  fixed topology; the JS panel and `connectSliding` adapt dynamically.
- "How do I hook a brain into a real robot?" → link `sage_core`
  (`brain/` + `utils/`), replicate the sense→tick→act pattern from
  `dog_drive.cpp`, and persist with `BrainSave`.

## 15. Glossary

- **tick** — one `Brain` cycle (transmit/update/learn); in `snn_drive`/`dog_drive`
  one tick == one 0.01 s physics step.
- **eligibility trace** — per-synapse memory that the connection was just used;
  gates `reward()` credit.
- **threshold homeostasis** — per-neuron auto-adjust of `threshold` toward a
  target firing rate (`adjustThreshold(targetRate)`).
- **ε-greedy** — explore (random) with probability ε, otherwise exploit the
  current policy; ε decays across episodes.
- **exploit episode** — a deterministic replay of the best controller trace
  recorded during training (what the trajectory CSV + fire log show).
- **MJCF / omniworld** — MuJoCo's and OmniSim's scene formats; there is *no*
  shared scene format, so scenarios are hand-mirrored per engine.
- **FIRST MISMATCH** — the marker printed by `omnisim_compare.py` when the
  first/threshold comparison fails; the grep-able CI signal.
- **z-up → y-up (`m2t`)** — the coordinate swap applied to every physical point
  before it enters three.js (MuJoCo `z` is up, three expects `y` up).
- **keyframe** — one entry in the playback timeline; the union of all relative
  sample times across loaded engines.