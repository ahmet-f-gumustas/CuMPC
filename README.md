# CuMPC — cuMPC-Mujoco

**CuMPC**, differential-drive bir tarım robotunu cm hassasiyetinde referans takip ettiren;
obstacle- ve terrain-aware; GPU-accelerated bir **MPPI (Model Predictive Path Integral)**
kontrolcüsüdür. Bu repo, **gerçek custom CUDA çekirdeğinin** (cuRAND sampling +
thread-per-rollout + reduction) **MuJoCo** simülasyon kabuğu içinde geliştirilmesidir —
çekirdek production cuMPC'ye (ROS 2 / gerçek robot) **olduğu gibi** taşınır.

Proje spec'i: [cuMPC-Mujoco.md](cuMPC-Mujoco.md) · Sonuçlar: [docs/RESULTS.md](docs/RESULTS.md)
· Phase 0 raporu: [docs/PHASE0_REPORT.md](docs/PHASE0_REPORT.md)

## Mimari

- **CUDA çekirdek** (`core/`, `dynamics/`, `perception/*.cuh`): sampling → rollout
  (exact-arc diff-drive + slip + cost) → min/weights/weighted-update reduction →
  warm-start. RAII device buffer'lar, deterministik reduction, `compute-sanitizer` temiz.
- **pybind11 köprüsü** (`bindings/`): `MPPIController.step(state, ref_window) → [v, omega]`,
  `set_esdf` / `set_elevation` upload'ları. PyTorch yok.
- **MuJoCo harness** (`mujoco_harness/`): yalnız dış kabuk — sensör okuma (framepos/quat,
  IMU, encoder, rangefinder fan), referans pencere hazırlama, aktüatör dönüşümü, loglama.
  Hiçbir kontrol matematiği Python'da değildir.

## Sonuç özeti (RTX 4070, K=2048, H=40)

| Metrik | Hedef | Ölçülen |
|---|---|---|
| Cross-track RMS (düz / eğri) | < 0.03 m | 0.008 / 0.028 m |
| Cross-track RMS (S-curve) | < 0.05 m | 0.042 m |
| Collision (engelli kurs) | 0 | 0 |
| MPPI loop-rate | ≥ 20 Hz | ~4500-6000 Hz (step ≈ 165 µs) |
| Slip-aware vs naive (yamaç) | daha düşük RMS | 0.096 vs 0.128 m (−%25) |
| compute-sanitizer (4 araç) | temiz | 0 hata / 0 hazard |

## Kurulum

```bash
uv venv --python 3.10 && source .venv/bin/activate
uv pip install mujoco numpy pyyaml matplotlib pybind11 scikit-build-core "cmake>=3.24" ninja pytest
uv pip install -e .        # CUDA core + pybind11 modülü (CMake, sm_89)
```

## Çalıştırma

```bash
MUJOCO_GL=egl python examples/drive_manual.py        # M0: manuel sürüş
MUJOCO_GL=egl python examples/track_straight.py      # M1 (+ track_curve / track_scurve)
MUJOCO_GL=egl python examples/full_course.py --no-terrain   # M2: engel
MUJOCO_GL=egl python examples/full_course.py                # M3: engel + terrain
MUJOCO_GL=egl python examples/slope_traverse.py      # M3: slip-aware vs naive
python metrics/evaluate.py --all                     # metrikler
python metrics/plot_results.py --all                 # docs/plots/*.png
```

GUI için `--viewer` bayrağını ekleyin (`MUJOCO_GL` gerekmez).

## Test / doğruluk

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest test/ -q          # 22 test
MUJOCO_GL=egl compute-sanitizer --tool memcheck python examples/track_scurve.py
```

## Kapsam

Bu repo spec gereği ROS 2, gerçek donanım, GP/ensemble, CVaR ve evidential DL içermez —
bunlar production cuMPC kabuğunun konusudur. Burada üretilen kernel'ler, query'ler ve
naming production ile birebir aynıdır.

## Lisans

GNU General Public License v3.0 — bkz. [LICENSE](LICENSE).
