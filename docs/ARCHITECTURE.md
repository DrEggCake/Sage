# Sage — Architecture & Developer Guide

A from-scratch explanation of the whole project: what every language,
directory, file, method, script, workflow and CLI is for, how the pieces talk
to each other, and how to run and extend everything.

---

## 1. What Sage is

Sage is an **experimental spiking-neural-network (SNN) system** built to be
embedded into embodied learning and robotics. It has three layers:

1. **The brain engine (C++)** — a faithful port of the `Sage-Java` SNN to
   modern C++ (`Brain` / `Neuron` / `Synapse`), plus JSON persistence.
2. **Embodied learning (C++ + Python)** — two simulator integrations that let a
   brain perceive an environment, produce actions, and learn from reward:
   - **MuJoCo** (DeepMind physics) via a C++ harness;
   - **OmniSim** (OmniLink's Newton "mujoco" physics) via an HTTP/JSON harness
     driven from a Python client.
3. **Replay / visualization (Python + JavaScript)** — a browser GUI that plays
   back recorded trajectories side-by-side in 3D and animates the SNN's firing
   live, with a web server that turns CSV artifacts into a manifest the browser
   can fetch.

Historical lineage: `Sage-Java` (https://github.com/DrEggCake/Sage-Java) is the
original; the SNN math in this repo is a direct port of it. This C++ repo is
the "embodied" branch: brain + physics + replay, wired for CI.

---

## 2. The four languages, and why each one is here

| Language | Where | Role |
|---|---|---|
| **C++20** | `brain/`, `utils/`, `main.cpp`, `tests/cross_sim_harness.cpp`, `tests/snn_drive.cpp`, `tests/brain_test.cpp` | The brain engine and every heavy/simulation-side path. C++ is used because the SNN core is the artifact you'd ship into a robot; spiking sims are hot loops that shouldn't pay interpreter overhead, and MuJoCo's official C API is C/++ anyway. |
| **Python 3** | `tests/omnisim_compare.py`, `tests/replay/serve_replay.py` | The *glue* layer. OmniSim's engine embeds its own system Python 3.12 and exposes an HTTP/JSON protocol, so a Python client is the natural way to talk to it. The replay server is a tiny HTTP static server whose only real job is to list CSVs on disk as JSON and serve files to the browser. Nothing in Python touches physics directly. |
| **JavaScript (+ HTML/CSS)** | `tests/replay/replay.html`, `tests/replay/vendor/*` | The replay *GUI*. It runs in any browser with zero build step, and WebGL (three.js) gives free 3D + orbit cameras for the trajectory views. The SNN panel redraws the 5-layer spiking net in SVG/DOM each tick. |
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
│   ├── omnisim_compare.py       # OmniSim HTTP client + comparison (Python)
│   ├── models/                  # MuJoCo MJCF XML: arm_reach/ball_drop/cart_pole
│   ├── omnisim_worlds/          # mirrored .omniworld scenes for OmniSim
│   ├── results/                 # output CSVs (git-ignored, runtime artifacts)
│   └── replay/                  # browser replay player
│       ├── replay.html          # the GUI (three.js + SNN panel)
│       ├── serve_replay.py      # local HTTP server + manifest generator
│       └── vendor/              # vendored three.js 0.160.0 + OrbitControls
├── main.cpp                     # demo/training CLI for the brain (epsilon-greedy)
├── CMakeLists.txt               # build: sage_core lib + 4 executables
├── .github/workflows/
│   └── cross-sim-test.yml       # CI: MuJoCo baselines + OmniSim build + compare
├── docs/
│   ├── ARCHITECTURE.md          # this file
│   └── deliverable_cross_sim_first_mismatch.md  # deliverable write-up
└── README.md
```

Git-ignored: `data/*` (trained brains + firing logs), `build/*`, `tests/results/*`,
`__pycache__/`, `*.pyc`. So `data/brain/` and `data/firelog/` are local-only
artifacts; committed tests must regenerate what they need.

---

## 4. Build system (CMake)

`CMakeLists.txt`:

- `FetchContent` pulls **nlohmann/json** (used only by `BrainSave`).
- **`sage_core`** — static lib from `brain/*.cpp` + `utils/BrainSave.cpp`. No
  external deps beyond JSON. This is the unit you'd drop into a robot.
- **`sage`** — `main.cpp` linked to `sage_core` (the epsilon-greedy trainer).
- **`brain_test`** — `tests/brain_test.cpp`, the engine unit checks.
- **`cross_sim_test`** and **`snn_drive`** — MuJoCo-gated. CMake looks for
  MuJoCo via `MUJOCO_DIR` (default `~/mujoco`): needs `include/mujoco/mujoco.h`
  and a `libmujoco` (`.so`/`.a`). If missing, both targets are skipped with a
  warning so the rest of the project still builds.

Build:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

After editing `CMakeLists.txt` re-run the configure step; existing build dirs
won't pick up new targets automatically.

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
| `RESTING_STRENGTH` | 0.05 | the baseline every weight relaxes toward |
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

> **Finding (read before predicting anything):** with `LEAK_RATE 0.005`/tick and
> reward scaled by a 0.02 learning rate, a sparse reward of ±1 per episode gives a
> weight step of ±0.02, while the leak over a 240-tick episode eats ~1.2. The
> reward signal is *erased* by the leak almost instantly. Empirically the saved
> weights stay pinned at 0.05. The two mechanisms that actually produce behavior:
> **(a) threshold homeostasis** (`Neuron::adjustThreshold`) and **(b) sustained
> co-firing Hebbian boosts**, which net *up* by `HEBB_BOOST − LEAK_RATE` each tick
> as long as a synfire path keeps iterating. Sparse, single-shot reward is a
> documented weakness of this learning scheme — see `snn_drive` results below.

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
  every `fireCount`.
- `reset()` — clears every neuron + synapse trace and dopamine. **Caveat:** it
  also zeroes `totalReward`; callers that need running totals accumulate reward
  on their own side (`main.cpp`, `snn_drive`).
- Wiring: `connectSliding(from, to, limit)` — for source `i`, connect to targets
  `start .. start+limit` where `start = floor(i·(N2−K)/(N1−1))`. Every source
  thus sees the same-sized receptive window; layer edges land densely on a
  diagonal band. (The JS network panel re-implements this identically so SVG
  edges match C++ synapses 1:1.)
- Learning-rate and topology accessors: `setLearningRate`, `getLearningRate`,
  `getLayerSizes`, `getWiringLimits`, `getTotalSteps`; introspection for
  persistence/replay: `getNeuronThresholds`, `getSynapseStrengths`,
  `getFiredFlags()` (an `std::vector<int>` of 0/1 per neuron, in layer order —
  what the replay panel colors by).

### 5.4 `utils/BrainSave`

Stateless JSON persistence with no file-locking assumptions.

- `BrainSave::save(brain, dir)` — writes `data/brain/brain_N.json` where `N` is
  the next free number; returns the path. Payload: topology, wiring limits,
  learning rate, every neuron's threshold, every synapse's strength.
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

### 6.1 Fire-log format (`main.cpp --fire-log` / `snn_drive --fire-log`)

A CSV plus a JSON sidecar, produced while training.

```
# topology 8,16,16,16,4
# episodes 300 decisionTicks 240 seed 12 lr 0.02 epsilonStart 1.0 epsilonEnd 0.05
ep,tick,i0,i1,...,f0,f1,...,f59
0,0,0.5,0.2,...,0,0,0,...
...
```

- Comment header lines start with `#`; the first says the topology, the second
  the run config.
- One row per *tick*; columns: episode, tick, the 8 input voltages, then one
  0/1 fired flag per neuron (input + hidden + output — total = full-layer count).
- `FILE.meta.json` sidecar: `{brain, layerSizes, wiringLimits, episodes,
  decisionTicks, seed, lr, epsilonStart, epsilonEnd, successes, totalReward,
  episodesTrained}`.
- `snn_drive` additionally emits a `# hz 100` line (ticks per real second) and
  records `stepsPerEpisode`, `shapingWeight`, `task` in its meta.

The replay engine derives neuron coordinates (layer col + index) from the
topology header and plays rows at `1/hz` per row when present.

---

## 7. MuJoCo side (C++)

### 7.1 Models (`tests/models/*.xml`, MJCF)

Three minimal worlds, each mirrored for OmniSim under `tests/omnisim_worlds/`:

- `cart_pole.xml` — a cart sliding on a floor with an inverted pole; the
  cart is pushed by an effort-capped motor (`control`). Physics dt 0.02 s.
- `ball_drop.xml` — a sphere falling under gravity onto a floor. dt 0.005 s.
- `arm_reach.xml` — a 2-DOF planar arm (shoulder + elbow) reaching for a
  colored target sphere. dt 0.01 s. This is the model the SNN drives.

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

### 7.3 `tests/snn_drive.cpp` (`snn_drive`) — SNN embodied learning

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
of all runs; final end-dist 0.24–0.85. Root cause = §5.2: sparse payoff is erased
by `LEAK_RATE`; the saved brain JSONs (`build/data/brain/brain_14.json` etc.)
show every synapse at 0.05. What *did* happen is a working closed loop — the
brain read MuJoCo, modulated motor outputs, and the replay shows correlated
firing + arm movement. **The integration works; convergence is a documented
limitation of the credit-assignment scheme, not a bug.** Sample artifact:
`build/data/snn_results/arm_snn_mujoco.csv` (150 episodes × 240 steps),
`build/data/firelog/arm_snn.log.csv`.

### 7.4 `tests/brain_test.cpp` (`brain_test`)

Small assert-driven unit suite (no framework): tick advances `totalSteps`,
setInput/getOutput don't throw, `reward` accumulates `totalReward`, `BrainSave`
roundtrip bit-preserves learningRate/topology/wiring/thresholds/strengths, and
`reset` clears transient state. Exit code 0 iff all pass.

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
within ~0.11). Documented at length in `docs/deliverable_cross_sim_first_mismatch.md`.

---

## 9. CI — GitHub Actions + the GitHub CLI

### 9.1 The workflow (`.github/workflows/cross-sim-test.yml`)

Runs on `ubuntu-24.04`, timeout 150 min. Two lanes:

1. **Cross-sim test lane** — mirrors OmniSim's own linux-build recipe:
   - Python must be **3.12** (checked explicitly; OmniSim's harness is
     python-version-sensitive);
   - apt installs X deps: `xvfb xauth libxcb-*0 mesa-vulkan-drivers` etc.;
   - OmniSim boots with `OMNISIM_SKIP_TORCH=1` (no neural libs needed),
     `OMNISIM_NO_WINDOW=1` (headless), plus `QT_QPA_PLATFORM=xcb` and
     `LIBGL_ALWAYS_SOFTWARE=1` for Qt/mesa under Xvfb;
   - **per-scenario isolated ports** — cart_pole `6811`, ball_drop `6862`,
     arm_reach `6921`, supervisor = scenario port + 10. Isolation prevents one
     stuck scenario from poisoning the others and keeps runs parallelizable;
   - builds this repo's MuJoCo targets against `MUJOCO_DIR=${{ github.workspace }}/.mujoco`,
     runs the three baselines (dropping `tests/results/*_mujoco.csv`), then
     launches `tests/omnisim_compare.py` per scenario against its port.
2. **Build+unit lane** — `sage_core` + `sage` + `brain_test` compile and
   `brain_test` passes, with artifacts uploaded (brains/firelogs/recordings via
   `actions/upload-artifact` so trained data can be pulled down).

**Trap:** the comparison step is `continue-on-error: true`, so the run still
*turns green* when the physics diverge. **Green ≠ success.** Always confirm with
`gh run view --log` and download the artifacts to read the actual numbers.

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

The local Arch box **can** build/run MuJoCo (`sage`, `brain_test`,
`cross_sim_test`, `snn_drive` all run locally) and the replay server, but
**cannot** run OmniSim — that only happens on CI (linux runner + system
Python 3.12 + Xvfb). So the local loop is: build → run MuJoCo targets → inspect
CSVs/replay; the CI loop adds: build OmniSim → load each world → compare servos.

---

## 10. The replay player (Python + JavaScript)

### 10.1 `tests/replay/serve_replay.py`

A stdlib HTTP server with three jobs:

- serve the static GUI (`replay.html`, `vendor/*`);
- serve recorded data: safety-scoped routes that resolve a filename under a
  configured root and return it (CSVs, JSON, bundled three.js);
- generate **`manifest.json`** — the index the player renders from.

`build_manifest(...)` scans configured roots and emits:
- `recordings[]` — id + file per MuJoCo/OmniSim CSV pair (same base name with
  `_mujoco` / `_omnisim` suffixes are glued into one playback entry),
  `simTickMs` sources, etc.;
- `firelogs[]` — `{id, file, rows, meta}` (meta = the sidecar described in §6.1);
- `brains[]` — `{id, file, layerSizes, wiringLimits, learningRate,
  episodesTrained, successes, totalReward, bestReward, totalSteps}`.

CLI flags: `--port` (default 8787), `--no-browser`, `--recordings DIR` (default
`tests/results`), `--firelogs DIR`, `--brains DIR`, `--root` for static files.
Run it and open `http://127.0.0.1:8787/` in a browser.

### 10.2 `replay.html` — the GUI

Three view tabs behind a side toolbar:

1. **Replay** — two synchronized 3D viewports (MuJoCo scene + OmniSim scene),
   each with a free OrbitControls camera. Track/step/loop controls:
   - Space = pause, `,`/`.` = framestep, `[`/`]` = speed 0.1–8×
     (speed multiplies *sim seconds*, so physics pacing survives replays),
     `L` = loop, `F` = fullscreen, `R` = reset camera, click-drag to orbit.
   - Four-corner HUD: (tl) scenario/samples/duration; (tr) time/keyframe/position/
     Δ/sim_ms/status; (bl) cross-engine distance + final positions; (br) controls
     legend. Start/end markers + trailing particle ribbon show the path.
2. **SNN** (`btn-net`, third toggle) — the network panel: an SVG of the 5 layers
   (8/16/16/16/4), one column per layer, layered top→bottom. Edge alpha ∝ synapse
   strength (the JS `connectSliding` re-implementation matches the C++ wiring, so
   the 168 edges equal the 168 synapse weights exactly). Per tick it colors fired
   neurons green-glow, renders the 8 input values as intensity bars, and ticks at
   the fire-log's `hz` (default 24/s) with `-`/`+`/loop/Play transport. Status
   line: `ep/tick/row/fired/brain`. Brain-only mode works with no recordings —
   the panel can animate a bare brain from a firelog.
3. Loading — a dropdown populates from the manifest (`recordings`/`firelogs` /
   `brains` selects); playback keeps the two 3D views and the SNN panel time-synced.

Key JS functions: `init()` (reads manifest, builds UI), `loadRecording(id)` /
`loadFireLog(id)` / `loadBrainNet(id)`, `rafLoop(dt)` (drives all views),
`frameStep(dt)` / `stepBrain(dt)`, `buildNet()` / `renderNetRow()`
(layer layout + per-tick spike coloring), `parseFireLog()` (topology + hz +
rows), and the three.js scene/camera/HUD wiring. It is dependency-light:
everything but three.js is hand-rolled (no framework, no build step).

---

## 11. Data formats, at a glance

| Artifact | Format | Columns / shape |
|---|---|---|
| MuJoCo trajectory | CSV | `time,x,y,z[,vx,vy,vz]` under `tests/results/{name}_mujoco.csv` |
| OmniSim trajectory | CSV | `time,sim_time_ms,x,y,z` under `tests/results/{name}_omnisim.csv` |
| Fire log | CSV | §6.1: `# topology/# episodes...`, then `ep,tick,i0..,f0..` |
| Brain (saved) | JSON | `brain_N.json`: topology, wiring, lr, thresholds, strengths |
| Fire-log sidecar | JSON | `FILE.meta.json`: run config + stats (see §6.1) |
| Replay manifest | JSON | `recordings[]`, `firelogs[]`, `brains[]` (see §10.1) |

Trajectory `time`/`sim_time_ms` were aligned at collection time so the player can
overlay both engines sample-for-sample.

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

# 6. serve the replay player
python3 tests/replay/serve_replay.py --no-browser --recordings tests/results \
    --firelogs data/firelog --brains data/brain --port 8787
# open http://127.0.0.1:8787/  → Replay tab (both engines) + SNN tab (firing)

# 7. (OmniSim side only runs on CI; locally it just prints MuJoCo-only counts)
python3 tests/omnisim_compare.py --harness-url http://127.0.0.1:6789
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
  failure noise). Details + traces in `docs/deliverable_cross_sim_first_mismatch.md`.
- **The SNN's reward path barely learns sparse tasks.** `LEAK_RATE` (0.005/tick)
  swamps one-shot `reward()` updates scaled by a 0.02 learning rate; real
  behavior change comes from threshold homeostasis and sustained Hebbian
  co-activation. `snn_drive` reaching therefore achieves 0/∞ — a documented
  limitation of credit assignment in this port, with the closed-loop integration
  itself validated.
- **Replay timing is approximate.** Sample times are normalized to episode
  relative time (`t - t0`): `sim ms` shown in the HUD is base + rel×1000; the
  SNN panel ticks at fire-log `hz` (default 24/s), so brain-vs-physics rate
  matches by definition only for `# hz`-tagged logs.
- **Green CI runs are not proof.** The comparison step is non-fatal by design;
  always read the numbers from logs/artifacts.

## 14. Where to look next (by question)

- "How do I add a new scenario?" → `tests/models/*.xml` (+ mirrored
  `tests/omnisim_worlds/`), add a `SCENARIOS` entry in `omnisim_compare.py`,
  and a branch in `cross_sim_harness.cpp`.
- "How do I make the SNN learn a task?" → `tests/snn_drive.cpp` (control
  mapping) held hostage by `Synapse::LEAK_RATE` / `Brain::reward` scaling;
  look at balancing leak vs learning rate before touching the task.
- "How do I change the network shape?" → the `Brain` constructor + its 5-layer
  fixed topology; the JS panel parses topology dynamically.
- "How do I hook a brain into a real robot?" → link `sage_core`
  (`brain/` + `utils/`), replicate the sense→tick→act pattern from
  `snn_drive.cpp`, and persist with `BrainSave`.

## 15. Glossary

- **tick** — one `Brain` cycle (transmit/update/learn); in `snn_drive` one tick
  == one 0.01 s physics step.
- **eligibility trace** — per-synapse memory that the connection was just used;
  gates `reward()` credit.
- **threshold homeostasis** — per-neuron auto-adjust of `threshold` toward a
  target firing rate (`adjustThreshold(targetRate)`).
- **ε-greedy** — explore (random) with probability ε, otherwise exploit the
  current policy; ε decays across episodes.
- **MJCF / omniworld** — MuJoCo's and OmniSim's scene formats; there is *no*
  shared scene format, so scenarios are hand-mirrored per engine.
- **FIRST MISMATCH** — the marker printed by `omnisim_compare.py` when the
  first/threshold comparison fails; the grep-able CI signal.