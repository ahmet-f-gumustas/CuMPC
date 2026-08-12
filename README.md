# CuMPC

**CuMPC** is a GPU-accelerated **MPPI (Model Predictive Path Integral)** controller
core for differential-drive mobile robots. It tracks a reference path with
centimeter-level accuracy while staying obstacle- and terrain-aware, and it is
written in **pure custom CUDA** — no PyTorch, no inference frameworks.

## Features

- **Custom CUDA pipeline** — cuRAND control sampling, one-thread-per-rollout
  cost evaluation, deterministic reductions, warm-started nominal plan
- **Exact-arc differential-drive dynamics** with slip awareness
  (`kappa_v`, `kappa_w`, lateral drift `v_y`)
- **Obstacle costs** from a signed-distance field (ESDF) queried on device with
  bilinear interpolation (hard barrier + soft margin)
- **Terrain costs** from an elevation feature map (slope, roughness,
  traversability) with a rollover guard
- **Deterministic by design** — no `atomicAdd` in reductions; a fixed seed
  reproduces results bit-exactly
- **RAII device memory** and error-checked CUDA calls throughout; clean under
  `compute-sanitizer` (memcheck, racecheck, initcheck, synccheck)
- **pybind11 bridge** — NumPy in, NumPy out

## How a step works

```
SAMPLE   U[k] = U_nom + eps,  eps ~ N(0, diag(sigma_v, sigma_omega)), hard clamp
ROLLOUT  one thread per sample k: H exact-arc dynamics steps + running cost
         (cross-track, heading, progress, smoothness, acceleration,
          ESDF obstacle, terrain, arc-length terminal)
REDUCE   rho = min S  →  w = exp(-(S - rho)/lambda)  →  weighted average → U_nom
OUTPUT   control = U_nom[0]; warm-start shift
```

All control math runs on the GPU. The host only uploads the reference window
and downloads the resulting `[v, omega]`.

## Performance

| Setup | Step latency |
|---|---|
| K=2048 samples, H=40 horizon, RTX 4070 Laptop GPU | **≈ 150 µs** (~6 kHz) |

That is 81,920 dynamics + cost evaluations per control step, leaving more than
two orders of magnitude of headroom over a typical 20 Hz control loop.

## Non-blocking use

`step()` submits the pipeline, waits for it and returns the control — convenient, and the reason it
was written that way. It is also, measured on an RTX 4070 Laptop at K=2048/H=40, **77 % wait**: the
8-byte device-to-host copy of the result is where the host stops and waits for every kernel queued
before it.

A caller that has something else to do meanwhile can split the two:

```cpp
MPPI planner(cfg);
planner.enqueue(state, reference, n);   // submits and returns   —  16.5 µs
// ... do other work here ...
while (!planner.poll()) { /* not ready yet; poll() never blocks */ }
const float2 control = planner.result();
```

Measured on the same machine and configuration: **submission 16.5 µs, wait 130.8 µs**. The GPU work
is unchanged — what the split recovers is the host's time, not the device's.

Everything runs on the controller's **own non-blocking stream** (`planner.stream()`), so it neither
serialises against nor is serialised by the legacy default stream. Nothing on the hot path allocates:
buffers keep their capacity across calls, and a cost map that is already on the device can be adopted
without a copy:

```cpp
planner.set_cost_grid_device(device_ptr, meta);   // no H2D, caller keeps ownership
```

## Requirements

- NVIDIA GPU and CUDA Toolkit ≥ 12.x (`nvcc` on PATH)
- Python ≥ 3.10
- CMake ≥ 3.24 and a C++20 compiler (fetched automatically when installing via pip)

The default build targets `sm_89` (RTX 40xx). For other GPUs:

```bash
CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=86" pip install ...   # e.g. RTX 30xx
```

## Installation

```bash
pip install "cumpc @ git+https://github.com/ahmet-f-gumustas/CuMPC.git"
```

or for development:

```bash
git clone https://github.com/ahmet-f-gumustas/CuMPC.git
cd CuMPC
pip install -e .
```

## Quick start

```python
import numpy as np
import cumpc_core as cc

cfg = cc.MPPIConfig()                          # K=2048, H=40, dt=0.05
cfg.lambda_, cfg.sigma_v, cfg.sigma_omega = 50.0, 0.15, 0.25

r = cfg.robot
r.wheel_radius, r.track_width = 0.10, 0.40
r.v_max, r.omega_max, r.a_max, r.alpha_max = 2.0, 2.0, 2.0, 3.0
r.robot_radius = 0.32

cfg.slip.kappa_v = cfg.slip.kappa_w = 1.0      # identity = no slip compensation

w = cfg.weights
w.w_lat, w.w_head, w.w_prog, w.w_du, w.w_term = 50.0, 10.0, 50.0, 0.1, 1000.0
w.w_coll_hard, w.w_coll_soft = 1e6, 200.0
w.w_slope, w.w_rough, w.w_trav, w.w_rollover = 20.0, 10.0, 30.0, 1e5

c = cfg.cost
c.k_accel, c.safe_hard, c.safe_soft = 0.1, 0.10, 0.30
c.rollover_slope, c.term_slack = 0.45, 2.2

ctrl = cc.MPPIController(cfg)

# optional: obstacle / terrain maps (host-built once, uploaded to the GPU)
# ctrl.set_esdf(sdf_grid, origin_x, origin_y, res)            # (ny, nx)
# ctrl.set_elevation(feature_grid, origin_x, origin_y, res)   # (ny, nx, 3)

state = np.zeros(3, dtype=np.float32)          # [px, py, theta]
xs = np.arange(64, dtype=np.float32) * 0.05    # reference window: straight ahead
ref = np.stack([xs, np.zeros_like(xs), np.zeros_like(xs)], axis=1)  # (N,3) x,y,heading

v, omega = ctrl.step(state, ref)
```

Utility functions: `build_esdf` (circles/boxes → SDF grid),
`build_elevation_features` (heights → slope/roughness/traversability),
`simulate_controls` (run the production dynamics on device),
`query_esdf` / `query_terrain` (debug map queries), `last_rollouts`,
`nominal`, `set_slip`, `reset`.

## Development Roadmap

**v0.1 — Core controller** *(current)*
- [x] MPPI step pipeline in pure CUDA (sample → rollout → reduce → warm start)
- [x] Exact-arc differential-drive dynamics with slip parameters
- [x] ESDF obstacle costs and terrain costs with rollover guard
- [x] Deterministic reductions, RAII device memory, sanitizer-clean
- [x] pybind11 / NumPy API

**v0.2 — Hardening**
- [ ] Publish the unit test suite (dynamics vs. analytic references, sampling
      statistics and determinism, bilinear map queries, controller smoke tests)
- [ ] Build CI (compile + host-side checks)
- [ ] Configurable CUDA architecture in packaging (default `native`)
- [ ] Runtime-tunable weights and temperature without rebuilding the controller

**v0.3 — Performance**
- [ ] Reference window in shared memory + local nearest-point search
- [ ] CUDA Graphs for the step pipeline (cut kernel launch overhead)
- [x] Async map uploads on a dedicated stream — with the buffer lifetime that makes them safe, an
      `enqueue`/`poll` split and a pinned result buffer
- [ ] Benchmarks at larger sample counts (K = 8k–16k)

**v1.0 — Deployment**
- [ ] ROS 2 integration shell (separate repository)
- [ ] State-dependent slip estimation driven by terrain queries
- [ ] Prebuilt wheels for common GPU architectures

## License

Apache License 2.0 — see [LICENSE](LICENSE).
