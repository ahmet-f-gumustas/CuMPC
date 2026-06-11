# cuMPC-Mujoco — Proje Spec'i (Claude Code-Ready)

> **Amaç:** cuMPC'nin **gerçek** CUDA çekirdeğini (cuRAND sampling + thread-per-rollout + reduction)
> MuJoCo simülasyon ortamında geliştirmek, test etmek ve profillemek. Burada üretilen CUDA kernelleri
> production cuMPC çekirdeğiyle **birebir aynıdır**; yalnızca dış kabuk MuJoCo (ROS2 değil) ve sensörler
> simülasyondur. Çekirdek daha sonra ROS2 / gerçek robot kabuğuna **olduğu gibi** taşınır.
>
> Bu dosya tek başına eksiksiz bir uygulama planıdır. Claude Code bunu baştan sona, milestone sırasıyla
> uygulamalıdır. **Phase 0 tamamlanmadan kod yazılmaz.**

---

## 0. Genel Bakış

cuMPC, differential-drive bir tarım robotunu cm hassasiyetinde referans takip ettiren; obstacle- ve
terrain-aware; GPU-accelerated bir **MPPI (Model Predictive Path Integral)** kontrolcüsüdür.

Bu projede MuJoCo, ROS2 + gerçek donanımın yerine geçen geliştirme ortamıdır:

- **ROS2 YOK** — Python (MuJoCo) ile doğrudan entegrasyon.
- **Gerçek donanım YOK** — MuJoCo sahnesindeki diff-drive robot + simülasyon sensörleri.
- **AMA MPPI çekirdeği gerçek custom CUDA'dır** (PyTorch değil). RTX 4070'te koşar, `compute-sanitizer`
  temiz, `nsys`/`ncu` ile profillenir.

MuJoCo fiziği tekerlek-zemin slip'ini doğal olarak ürettiği için slip compensation **otantik** test edilir
(rollout modeli slip-aware, ground-truth MuJoCo kendi slip'ini üretir → tracking farkı ölçülür).

### Kasıtlı kapsam sadeleştirmeleri
Production'a göre **dışarıda bırakılanlar**: GP / ensemble / learned residual, CVaR, evidential DL,
active perception, ROS2, gerçek donanım. **İçeride olan:** gerçek custom CUDA MPPI çekirdeği, exact-arc
diff-drive dynamics, slip-aware rollout, obstacle (ESDF) + terrain cost, RAII device buffers, pybind11
köprüsü.

---

## 1. Phase 0 / Önkoşul (Discovery)

> **Bu bölümdeki tüm kutular işaretlenmeden M0'a geçilmez.** Belirsiz teknik kararlar burada checkbox
> olarak bırakılmıştır; Claude Code bunları doğrular/seçer ve `docs/PHASE0_REPORT.md` olarak raporlar.

### 1.1 Toolchain doğrulama
- [ ] `nvcc --version` → CUDA Toolkit **>= 12.x** mevcut (production ile aynı major).
- [ ] `nvidia-smi` → RTX 4070 görünür, driver sürümü CUDA 12.x ile uyumlu.
- [ ] RTX 4070 compute capability **sm_89** (Ada Lovelace) doğrulandı → `CMAKE_CUDA_ARCHITECTURES=89`.
- [ ] `cmake --version` → **>= 3.24** (modern `CUDA_ARCHITECTURES` desteği).
- [ ] C++ derleyici **C++20** destekliyor (gcc >= 11 / clang >= 14).
- [ ] `pybind11` kurulabilir/erişilebilir (pip veya submodule).
- [ ] `compute-sanitizer` PATH'te (`compute-sanitizer --version`).
- [ ] `nsys --version` ve `ncu --version` mevcut (Nsight Systems + Compute).
- [ ] **PyTorch GEREKMİYOR** — çekirdek saf CUDA. Bağımlılık olarak eklenmeyecek (teyit).

### 1.2 Python ortamı
- [ ] `uv --version` mevcut; `uv venv` ile `.venv` oluşturulabiliyor.
- [ ] Python **3.10+** (`uv python pin 3.10`).
- [ ] `mujoco` python paketi kurulabiliyor (`uv pip install mujoco`), `import mujoco` çalışıyor.
- [ ] `mujoco.viewer` (passive viewer) bu makinede açılıyor (GUI veya `MUJOCO_GL=egl` ile headless).

### 1.3 Build entegrasyonu kararı (CUDA + pybind11 + uv)
- [ ] **KARAR:** Build backend olarak `scikit-build-core` (cmake'i `pyproject.toml`'a bağlar) **mı**,
      yoksa manuel `cmake` build + `.so`'yu pakete kopyalama **mı** kullanılacak?
      → Varsayılan öneri: **scikit-build-core** (uv ile temiz `uv pip install -e .`). Erişilemezse manuel
      cmake fallback.

### 1.4 Robot modeli erişilebilirliği (AÇIK KARAR)
- [ ] **KARAR:** MuJoCo diff-drive robot modeli olarak hangisi kullanılacak?
  - [ ] Seçenek A — **Minimal custom diff-drive MJCF** (2 tahrikli tekerlek + caster + box şasi).
        *Kabul edilebilir fallback; bağımlılık yok, tam kontrol.* → **Varsayılan tercih.**
  - [ ] Seçenek B — Clearpath **Jackal / Husky** community MuJoCo modeli (erişilebilirse).
  - Seçim kriteri: erişim + sensör site'larının kolayca eklenebilmesi. Erişim doğrulanamazsa **A** seçilir.

### 1.5 Perception kaynağı kararı (M2/M3)
- [ ] **KARAR:** Obstacle/terrain haritaları ilk aşamada **privileged bilgiden** mi build edilecek?
      → Varsayılan: **Evet** — ESDF, MuJoCo geom konumlarından; elevation, `hfield` verisinden (CPU build,
      device'a upload). **Rollout kernelindeki QUERY production ile birebir aynı** (build basit, query
      production). Depth/rangefinder'dan harita build'i M4 sonrası opsiyonel uzatma.

### 1.6 Perception sensör kararı
- [ ] **KARAR:** Engel/terrain algısı için MuJoCo'da **rangefinder ray fan** mı, **depth camera** mı?
      → Varsayılan: **rangefinder ray fan** (basit, deterministik). Depth camera (`mujoco.Renderer` depth)
      opsiyonel. *Not: M2/M3 maps privileged bilgiden geldiği için sensör bu milestone'larda yalnızca
      "gerçekçilik gösterimi"; harita build'in sensöre taşınması M4 sonrası.*

**Phase 0 DoD:** Yukarıdaki tüm kutular işaretli; `docs/PHASE0_REPORT.md` araç sürümleri + seçilen
kararları içeriyor.

---

## 2. Mimari

```
        ┌─────────────────────────────────────────┐
        │  mujoco_harness/ (Python)                │
        │  her step:                               │
        │   1) MuJoCo sensörlerini oku             │
        │      (framepos, imu, wheel vel, ranges)  │
        │   2) reference window'u hazırla          │
        │   3) controller.step(state, ref) çağır   │ ──┐
        │   4) dönen [v, omega]'yı aktüatöre uygula │   │
        │      (diff-drive → wheel velocities)     │   │
        └─────────────────────────────────────────┘   │
                          ▲                             │ pybind11
                          │ control [v, omega]          ▼
        ┌─────────────────────────────────────────────────────────┐
        │  CUDA cuMPC core (C++20 + CUDA, PRODUCTION-IDENTICAL)     │
        │   sample_controls → rollout_cost → reduce/weights/update  │
        │   diff_drive_step (exact arc + slip)                     │
        │   esdf.query / elev.query_terrain (device, production)    │
        │   RAII device buffers                                    │
        └─────────────────────────────────────────────────────────┘
```

- **Sınır netliği:** Python harness yalnızca *dış kabuktur* (sensör I/O + aktüatör + viz). Hiçbir kontrol
  matematiği Python'da yapılmaz. Tüm MPPI, dynamics, cost, query **CUDA tarafında**.
- Bu sınır, ROS2 kabuğuna geçişte yalnızca harness'in değişmesini (çekirdeğin sabit kalmasını) garanti eder.

---

## 3. Repo / Dosya Yapısı

```
cuMPC-Mujoco/
├── CLAUDE.md                     # Claude Code workflow + invariants (bu dosyadan türetilir)
├── README.md
├── pyproject.toml                # uv + scikit-build-core (veya manuel cmake notu)
├── CMakeLists.txt                # CUDA core + pybind11 modülü
│
├── core/                         # === GERÇEK CUDA ÇEKİRDEK ===
│   ├── sampling.cuh / sampling.cu      # cuRAND control sampling, clamp-to-limits
│   ├── rollout.cuh  / rollout.cu       # thread-per-rollout (K×H) + device cost
│   ├── reduction.cuh / reduction.cu    # min-cost (rho), weights=exp(-(1/λ)(S-ρ)), normalize, weighted sum
│   ├── mppi.cuh     / mppi.cu          # step() orkestrasyonu + warm-start
│   ├── device_buffer.cuh               # RAII device buffer (cudaMalloc/cudaFree)
│   ├── types.cuh                       # state/control/param/cost struct'ları (NAMING ZORUNLU)
│   └── cuda_check.cuh                  # CUDA_CHECK makrosu
│
├── dynamics/
│   └── diff_drive.cuh            # __device__ diff_drive_step: EXACT ARC + slip (kappa_v/kappa_w/v_y)
│
├── perception/
│   ├── map_view.cuh              # MapView + __device__ query (bilinear, ESDF, terrain) — PRODUCTION-IDENTICAL
│   ├── costmap.h / costmap.cpp   # ESDF build (privileged geom → host grid → device upload)
│   └── elevation.h / elevation.cpp  # terrain features (slope/roughness/trav) build → device upload
│
├── bindings/
│   └── pybind_module.cpp         # MPPIController sınıfı, NumPy I/O, set_esdf/set_elevation
│
├── mujoco_harness/
│   ├── scene.xml                 # MJCF: robot + heightfield terrain + obstacles + sensors + actuators
│   ├── robot_diffdrive.xml       # (custom seçilirse) include edilen robot tanımı
│   ├── sensors.py                # sensör okuma yardımcıları (state, imu, encoders, ranges)
│   ├── reference_path.py         # düz + eğri + S-curve path üretimi, resample, window
│   ├── controls.py               # [v, omega] → wheel velocity dönüşümü + clamp
│   └── run_sim.py                # ana run loop (sim ↔ controller)
│
├── config/
│   └── default.yaml              # K,H,dt,lambda,sigma_*; tüm cost ağırlıkları; robot params
│
├── examples/
│   ├── drive_manual.py           # M0: klavye/scripted manuel sürüş
│   ├── track_straight.py         # M1
│   ├── track_curve.py            # M1
│   ├── track_scurve.py           # M1 (+ M2/M3'te engel/terrain açık)
│   └── full_course.py            # M2+M3: engel + terrain birlikte
│
├── metrics/
│   ├── evaluate.py               # cross-track RMS, collision sayısı, loop-rate (Hz)
│   └── plot_results.py           # trajeler, hata zaman serisi, hız profili
│
├── test/
│   ├── test_dynamics.py          # exact-arc doğrulama (analitik referansa karşı)
│   ├── test_sampling.py          # noise dağılımı, clamp, determinizm (seed)
│   ├── test_query.py             # ESDF/terrain bilinear query doğruluğu
│   └── test_mppi.py              # tek-adım step() smoke + boyut/sınır kontrolleri
│
└── docs/
    ├── PHASE0_REPORT.md          # Phase 0 çıktısı
    └── RESULTS.md                # M4 metrik + profiling sonuçları
```

---

## 4. Teknoloji Stack

| Katman | Teknoloji |
|---|---|
| Çekirdek | **C++20 + CUDA (>= 12.x)**, cuRAND |
| Köprü | **pybind11** |
| Harness | **Python 3.10+**, `mujoco` (python bindings) |
| Paket/ortam | **uv** (+ `scikit-build-core` build backend) |
| Build | **CMake (>= 3.24)**, `CMAKE_CUDA_ARCHITECTURES=89` |
| Profiling | `compute-sanitizer`, `nsys`, `ncu` |
| YOK | **ROS2 yok, PyTorch yok** |

---

## 5. Naming Conventions (ZORUNLU — production cuMPC'ye birebir taşıma)

> Bu isimler **değiştirilemez**. `core/types.cuh` ve `config/default.yaml` bu isimlerle yazılır.

**State & control**
- `state = [px, py, theta]`  (theta: heading [rad])
- `control = [v, omega]`     (v: lineer hız [m/s], omega: açısal hız [rad/s])
- Integration: **EXACT ARC** (Euler **değil**).

**Slip parametreleri**
- `kappa_v`  → longitudinal slip faktörü (1.0 = slip yok)
- `kappa_w`  → angular slip faktörü (1.0 = slip yok)
- `v_y`      → lateral drift hızı [m/s] (side-slip)

**Cost ağırlıkları**
- `w_lat`        → cross-track
- `w_head`       → heading error
- `w_prog`       → progress (ileri ilerleme ödülü)
- `w_du`         → Δu smoothness
- `w_coll_hard`  → obstacle hard barrier
- `w_coll_soft`  → obstacle soft margin
- `w_slope`      → terrain slope
- `w_rough`      → terrain roughness
- `w_trav`       → traversability (düşük trav cezalandırılır)
- `w_rollover`   → rollover guard

**MPPI parametreleri**
- `K`            → num_samples
- `H`            → horizon
- `dt`           → rollout timestep
- `lambda`       → temperature
- `sigma_v`      → control noise std (v)
- `sigma_omega`  → control noise std (omega)

---

## 6. CUDA cuMPC Core Tasarımı

### 6.1 MPPI algoritması (her `step`)

```
Girdi:  state x0=[px,py,theta], reference window R, esdf, elevation, U_nom (warm-start)
1. SAMPLE:   U[k,t] = U_nom[t] + eps[k,t],  eps ~ N(0, diag(sigma_v, sigma_omega))
             U[k,t] clamp → [-v_max,v_max] × [-omega_max,omega_max]      (sampling kernel, cuRAND)
2. ROLLOUT:  her k için x0'dan H adım diff_drive_step ile ilerle;
             her adımda step_cost ekle → S[k]                              (rollout kernel, thread-per-rollout)
3. BASELINE: rho = min_k S[k]                                              (reduction: min)
4. WEIGHTS:  wgt[k] = exp(-(1/lambda) * (S[k] - rho))
             eta    = sum_k wgt[k]                                         (reduction: sum)
5. UPDATE:   U_nom[t] = (1/eta) * sum_k wgt[k] * U[k,t],  ∀t∈[0,H)         (weighted reduction over k)
6. OUTPUT:   control = U_nom[0]
7. WARMSTART:U_nom[0..H-2] ← U_nom[1..H-1];  U_nom[H-1] ← U_nom[H-2]
Çıktı: control=[v,omega]
```

`rho` (min-cost baseline) **sayısal taşmayı önler**: `exp(-(1/λ)(S-ρ))` daima ≤ 1.

### 6.2 Kernel'ler (signature kontratı)

```cpp
// core/sampling.cu — cuRAND ile K×H control örnekleme + clamp
__global__ void sample_controls_kernel(
    const float* __restrict__ U_nom,   // [H*2]
    float*       __restrict__ U,        // [K*H*2]  (out)
    curandState* __restrict__ rng,      // [K]  (persistent, seed'li)
    float sigma_v, float sigma_omega,
    float v_max, float omega_max,
    int K, int H);

// core/rollout.cu — thread-per-rollout: 1 thread = 1 tam rollout
__global__ void rollout_cost_kernel(
    const float* __restrict__ U,        // [K*H*2]
    const float3              x0,       // [px,py,theta]
    float*       __restrict__ S,        // [K]  (out: toplam cost)
    RobotParams  rp, SlipParams slip, CostWeights w,
    RefWindow    ref,                   // device pointer + meta
    MapView      esdf, MapView elev,
    float dt, int K, int H);

// core/reduction.cu
__global__ void reduce_min_kernel(const float* S, float* rho, int K);     // blok-reduce + atomicMin (float)
__global__ void weights_kernel(const float* S, float rho, float lambda,
                               float* wgt, float* eta_partial, int K);     // exp + blok-sum
__global__ void weighted_update_kernel(const float* U, const float* wgt,
                               float eta, float* U_nom, int K, int H);     // ∀(t,dim): Σ_k wgt[k]·U[k,t,dim]
```

**Thread organizasyonu**
- `sample_controls_kernel`: grid = K thread (her thread bir k için H×2 örnek üretir; kendi `curandState`'i).
- `rollout_cost_kernel`: **grid = K thread**, her thread H adımlık rollout + cost akümülasyonu (registers'da
  `px,py,theta`, `v_prev,omega_prev`). Global okuma yalnız `U[k,·]`; map query read-only.
- Reduction: standart blok-reduce; min için float `atomicMin` (bit-trick) veya iki aşamalı reduce.
- `weighted_update_kernel`: `(t,dim)` başına bir thread (toplam `H*2` thread/blok grid), k üzerinde döngü;
  veya transpose'lu reduce. **K büyük (≥2048) → coalesced erişim için layout `[t][dim][k]` veya `[k][t][dim]`
  kararını Phase 0 mikro-benchmark ile netleştir** (varsayılan `[k][t*2+dim]`).

**Determinizm:** seed config'ten; `curandState` dizisi bir kez init (`curand_init`), her step ileri sürülür.
Test'te sabit seed ile tekrarlanabilir sonuç.

### 6.3 Diff-Drive Device Dynamics (EXACT ARC + slip) — `dynamics/diff_drive.cuh`

```cpp
#pragma once
#include <cuda_runtime.h>

struct RobotParams {
    float wheel_radius;     // [m]
    float track_width;      // [m]
    float v_max, omega_max; // [m/s], [rad/s]
    float a_max, alpha_max; // [m/s^2], [rad/s^2]
    float robot_radius;     // [m] collision yarıçapı
};

struct SlipParams {
    float kappa_v;          // longitudinal slip faktörü (1.0 = slip yok)
    float kappa_w;          // angular slip faktörü
    float v_y;              // lateral drift [m/s]
};

// state=[px,py,theta], control=[v,omega] — EXACT ARC integration (Euler DEĞİL), slip-aware.
__device__ __forceinline__
void diff_drive_step(float& px, float& py, float& theta,
                     float v, float omega,
                     const SlipParams& slip, float dt)
{
    const float v_eff = slip.kappa_v * v;     // longitudinal slip
    const float w_eff = slip.kappa_w * omega; // angular slip
    const float th0   = theta;

    if (fabsf(w_eff) < 1e-5f) {               // düz çizgi limiti
        px   += v_eff * dt * cosf(th0);
        py   += v_eff * dt * sinf(th0);
        theta = th0;
    } else {                                  // EXACT ARC (sabit-curvature yay)
        const float R   = v_eff / w_eff;
        const float th1 = th0 + w_eff * dt;
        px   += R * (sinf(th1) - sinf(th0));
        py   += -R * (cosf(th1) - cosf(th0));
        theta = th1;
    }
    // lateral drift (side-slip): heading'e dik, +v_y = sola
    px += -slip.v_y * dt * sinf(th0);
    py +=  slip.v_y * dt * cosf(th0);
}
```

> **M1'de** slip kapalı: `kappa_v = kappa_w = 1.0`, `v_y = 0` (identity). **M3'te** slip değerleri terrain
> map query'sinden (traversability/slope → kappa tahmini) veya sabit estimate'ten beslenir. Kernel kodu
> aynı kalır; yalnız `SlipParams` kaynağı değişir.

### 6.4 Cost fonksiyonu (rollout kernelinde, device) — referans kontrat

```cpp
struct CostWeights {
    float w_lat, w_head, w_prog, w_du;
    float w_coll_hard, w_coll_soft;
    float w_slope, w_rough, w_trav, w_rollover;
};

// Tek rollout adımının cost'u. ref: o adıma karşılık gelen en yakın referans nokta+tangent.
__device__ __forceinline__
float step_cost(float px, float py, float theta,
                float v, float omega, float v_prev, float omega_prev,
                const RefPoint& ref, const CostWeights& w,
                const RobotParams& rp, const MapView& esdf, const MapView& elev,
                float dt)
{
    float c = 0.0f;
    const float ex = px - ref.x, ey = py - ref.y;

    // 1) cross-track (path'e dik mesafe)
    const float cross = -sinf(ref.heading) * ex + cosf(ref.heading) * ey;
    c += w.w_lat * cross * cross;

    // 2) heading error (wrap)
    float dpsi = theta - ref.heading;
    dpsi = atan2f(sinf(dpsi), cosf(dpsi));
    c += w.w_head * dpsi * dpsi;

    // 3) progress: path tangent yönünde ileri ilerleme ödülü
    c += -w.w_prog * (v * dt * cosf(dpsi));

    // 4) Δu smoothness
    const float dv = v - v_prev, dw = omega - omega_prev;
    c += w.w_du * (dv*dv + dw*dw);

    // 5) accel limit (soft) — control limiti sampling'de hard-clamp edilir
    c += accel_pen(dv / dt, rp.a_max);
    c += accel_pen(dw / dt, rp.alpha_max);

    // 6) obstacle — ESDF query (PRODUCTION-IDENTICAL)
    const float d = esdf.query(px, py);                 // signed distance, engele [m]
    const float d_clear = d - rp.robot_radius;
    if (d_clear < SAFE_HARD)  c += w.w_coll_hard;        // hard barrier
    if (d_clear < SAFE_SOFT) { float m = SAFE_SOFT - d_clear; c += w.w_coll_soft * m * m; } // soft margin

    // 7) terrain — elevation query (PRODUCTION-IDENTICAL)
    TerrainCell t = elev.query_terrain(px, py);          // {slope, roughness, traversability}
    c += w.w_slope * t.slope * t.slope;
    c += w.w_rough * t.roughness;
    c += w.w_trav  * (1.0f - t.traversability);
    if (t.slope > ROLLOVER_SLOPE) c += w.w_rollover;     // rollover guard (slope eşiği)

    return c;
}
```

Notlar:
- **Control limitleri** sampling kernelinde **hard clamp** (`[-v_max,v_max]×[-omega_max,omega_max]`); cost'ta
  ek olarak yumuşak bir `clamp_pen` opsiyonel.
- **Accel limitleri** soft penalty (`accel_pen` = limit aşımının karesi; sabit iç katsayı `K_ACCEL`,
  config'te `k_accel`).
- `SAFE_HARD`, `SAFE_SOFT`, `ROLLOVER_SLOPE` config'te (`safe_hard`, `safe_soft`, `rollover_slope`).
- Rollout sonunda toplam `S[k] = Σ_t step_cost` (+ opsiyonel terminal cost: hedefe/lookahead'e mesafe).

### 6.5 Map Query — `perception/map_view.cuh` (PRODUCTION-IDENTICAL)

```cpp
struct MapMeta { float origin_x, origin_y, res; int nx, ny; };  // grid: world ↔ index

struct MapView {                         // device-side read-only görünüm
    const float* data;                   // ESDF: [ny*nx] (signed dist)
    MapMeta m;
    // bilinear interpolated query (production'da da bu)
    __device__ __forceinline__ float query(float wx, float wy) const {
        float gx = (wx - m.origin_x) / m.res;
        float gy = (wy - m.origin_y) / m.res;
        // clamp + bilinear (4-komşu)
        ...
    }
};

struct TerrainCell { float slope; float roughness; float traversability; };
// elevation MapView: data layout [ny*nx*3] (slope,rough,trav) veya 3 ayrı düzlem;
// query_terrain(wx,wy) bilinear → TerrainCell.
```

- **Build (host, basit):**
  - `costmap.cpp`: privileged MuJoCo geom konum/yarıçaplarından grid'e Euclidean signed distance (ESDF).
    Basit `O(nx·ny·obstacles)` brute-force kabul; sonra GPU'ya taşınabilir.
  - `elevation.cpp`: `hfield` yüksekliğinden numerik gradient → `slope`; lokal varyans → `roughness`;
    `traversability = f(slope, roughness)` (örn. eşik tabanlı).
- **Query (device, production):** yukarıdaki `query`/`query_terrain` rollout kernelinde **birebir**
  kullanılır. Build basit başlayıp GPU'ya taşınsa da **query değişmez** — taşınabilirlik garantisi budur.

### 6.6 RAII device buffers — `core/device_buffer.cuh`

```cpp
template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t n) { CUDA_CHECK(cudaMalloc(&ptr_, n*sizeof(T))); n_=n; }
    ~DeviceBuffer() { if (ptr_) cudaFree(ptr_); }
    DeviceBuffer(const DeviceBuffer&) = delete;             // kopyalama yok
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&&) noexcept;                  // move OK
    T* get() const { return ptr_; }
    size_t size() const { return n_; }
    void upload(const T* host, size_t n);                   // cudaMemcpy H2D
    void download(T* host, size_t n) const;                 // D2H
private:
    T* ptr_ = nullptr; size_t n_ = 0;
};
```

- Tüm device buffer'lar (`U`, `U_nom`, `S`, `wgt`, `rng`, `esdf`, `elev`) `MPPIController` içinde
  `DeviceBuffer` üyesi olarak tutulur → leak yok, exception-safe.
- `cuda_check.cuh`: `CUDA_CHECK(call)` makrosu hatayı dosya/satır ile fırlatır.

### 6.7 pybind11 arayüzü — `bindings/pybind_module.cpp`

```cpp
class MPPIController {
public:
    explicit MPPIController(const MPPIConfig& cfg);                 // tüm param + buffer init + rng seed
    // x: (3,) [px,py,theta]; ref: (N,3) [x,y,heading] window; → (2,) [v,omega]
    py::array_t<float> step(py::array_t<float> x, py::array_t<float> ref);
    void set_esdf(py::array_t<float> grid, float ox, float oy, float res);        // H2D upload
    void set_elevation(py::array_t<float> grid, float ox, float oy, float res);   // H2D upload
    py::array_t<float> last_rollouts();   // (K,H,3) debug/viz (opsiyonel, download)
    py::array_t<float> nominal();         // (H,2) warm-start görünür
    void reset();                         // warm-start sıfırla
};

PYBIND11_MODULE(cumpc_core, m) {
    py::class_<MPPIConfig>(m, "MPPIConfig")...;   // K,H,dt,lambda,sigma_*, RobotParams, CostWeights, SlipParams
    py::class_<MPPIController>(m, "MPPIController")
        .def(py::init<const MPPIConfig&>())
        .def("step", &MPPIController::step)
        .def("set_esdf", &MPPIController::set_esdf)
        .def("set_elevation", &MPPIController::set_elevation)
        .def("last_rollouts", &MPPIController::last_rollouts)
        .def("reset", &MPPIController::reset);
}
```

**Reference window:** Harness, robotun path üzerindeki en yakın index'ini bulur (`s0`), oradan ileri
`N = H + margin` noktayı **sabit arc-length** ile resample edip `step`'e verir. Rollout kernelinde her
adım için referans noktası **window içinde en yakın** aramayla (küçük N, ör. 64) atanır → device'da
`O(K·H·N)` (≈ 2048·40·64 ≈ 5.2M) ucuz. *(Alternatif: arc-length progression; varsayılan windowed-nearest.)*

---

## 7. MuJoCo Harness

### 7.1 Sahne — `mujoco_harness/scene.xml`

İçerik:
- **Robot:** Phase 0 kararına göre custom diff-drive (varsayılan) veya Jackal/Husky.
- **Terrain:** `<hfield>` (eğimler + pürüz). `nrow/ncol`, `size` config'ten türetilir.
- **Obstacle(lar):** statik geom'ler (cylinder/box), konumları course'a göre.
- **Reference path:** kod tarafında üretilir (sahnede görselleştirme için site/geom marker opsiyonel).

Custom diff-drive MJCF iskeleti (Seçenek A):

```xml
<mujoco model="cumpc_diffdrive">
  <option timestep="0.002" integrator="implicitfast"/>
  <asset>
    <hfield name="terrain" nrow="128" ncol="128" size="15 15 0.6 0.05"/>
  </asset>
  <worldbody>
    <geom name="ground" type="hfield" hfield="terrain" pos="0 0 0" friction="1.0 0.05 0.001"/>
    <body name="chassis" pos="0 0 0.12">
      <freejoint name="base"/>
      <geom type="box" size="0.25 0.18 0.05" mass="10"/>
      <site name="imu_site" pos="0 0 0"/>
      <body name="left_wheel" pos="0 0.20 -0.05">
        <joint name="left_wheel" type="hinge" axis="0 1 0"/>
        <geom type="cylinder" size="0.10 0.04" mass="1.0" friction="1.2 0.05 0.001"/>
      </body>
      <body name="right_wheel" pos="0 -0.20 -0.05">
        <joint name="right_wheel" type="hinge" axis="0 1 0"/>
        <geom type="cylinder" size="0.10 0.04" mass="1.0" friction="1.2 0.05 0.001"/>
      </body>
      <body name="caster" pos="-0.22 0 -0.07">
        <geom type="sphere" size="0.05" mass="0.2" friction="0.1 0.01 0.0001"/>
      </body>
      <!-- rangefinder ray fan (Phase 0 kararı) -->
      <site name="lidar" pos="0.25 0 0.05"/>
    </body>
    <geom name="obs1" type="cylinder" pos="4 0.4 0.3" size="0.3 0.3" rgba="0.8 0.2 0.2 1"/>
    <geom name="obs2" type="box"      pos="7 -0.5 0.3" size="0.3 0.3 0.3" rgba="0.8 0.2 0.2 1"/>
  </worldbody>
  <actuator>
    <velocity name="left_motor"  joint="left_wheel"  kv="20"/>
    <velocity name="right_motor" joint="right_wheel" kv="20"/>
  </actuator>
  <sensor>
    <framepos  name="base_pos"  objtype="body" objname="chassis"/>   <!-- GPS/RTK proxy (ground-truth) -->
    <framequat name="base_quat" objtype="body" objname="chassis"/>
    <accelerometer name="imu_acc"  site="imu_site"/>
    <gyro          name="imu_gyro" site="imu_site"/>
    <jointvel name="lw_vel" joint="left_wheel"/>
    <jointvel name="rw_vel" joint="right_wheel"/>
    <!-- rangefinder ray fan: birden çok rangefinder veya replicate -->
    <rangefinder name="ray_c" site="lidar"/>
  </sensor>
</mujoco>
```

### 7.2 Sensörler — `mujoco_harness/sensors.py`
- **Position (GPS/RTK proxy):** `framepos` → ground-truth `px, py`; `framequat` → `theta` (yaw). Opsiyonel
  Gaussian gürültü (config `pos_noise_std`).
- **IMU:** `accelerometer` + `gyro` (+ `framequat`). M0/M1'de yalnızca loglanır; kontrol state'i framepos'tan.
- **Wheel encoders:** `jointvel`/`jointpos` (`lw_vel`, `rw_vel`) → odometry karşılaştırması (opsiyonel).
- **Perception:** rangefinder ray fan (Phase 0). M2/M3'te maps **privileged** bilgiden geldiği için ray
  okuması bu aşamada gerçekçilik gösterimidir.

### 7.3 Reference path — `mujoco_harness/reference_path.py`
- Üç segment birleşik: **düz** → **eğri (arc)** → **S-curve** (iki ters arc).
- Dense resample (sabit arc-length `ds`), her noktada `heading = atan2(dy, dx)`.
- `nearest_index(px,py)` ve `window(idx, N)` yardımcıları → `step`'e verilecek `(N,3)` pencere.

### 7.4 Run loop — `mujoco_harness/run_sim.py`
```
model, data = load(scene.xml); ctrl = MPPIController(MPPIConfig(**yaml))
ctrl.set_esdf(...); ctrl.set_elevation(...)        # M2/M3'te privileged build
path = build_reference_path(cfg)
while not done:
    state = read_state(data)                       # framepos+quat → [px,py,theta]
    idx   = path.nearest_index(state)
    refw  = path.window(idx, N)
    t0 = perf_counter()
    v, omega = ctrl.step(state, refw)              # <-- TÜM kontrol burada (CUDA)
    loop_dt = perf_counter() - t0                  # loop-rate metriği
    wl, wr = control_to_wheels(v, omega, cfg)      # diff-drive ters dönüşüm + clamp
    data.ctrl[:] = [wl, wr]
    mujoco.mj_step(model, data)                    # birden çok sim adımı / kontrol adımı (dt eşlemesi)
    log(state, [v,omega], loop_dt)
```
> **dt eşlemesi:** kontrol periyodu `dt` (config), MuJoCo `timestep` daha küçük (ör. 0.002). Her kontrol
> adımında `round(dt / timestep)` kez `mj_step` çağrılır.

### 7.5 Control → aktüatör — `mujoco_harness/controls.py`
```
v_left  = v - omega * track_width / 2
v_right = v + omega * track_width / 2
w_left  = clamp(v_left  / wheel_radius, ±w_wheel_max)
w_right = clamp(v_right / wheel_radius, ±w_wheel_max)
```
(velocity aktüatör hedefleri: `w_left`, `w_right`).

---

## 8. Config — `config/default.yaml`

```yaml
# === MPPI ===
mppi:
  K: 2048            # num_samples
  H: 40              # horizon
  dt: 0.05           # rollout/control timestep [s]  (≈ 20 Hz)
  lambda: 1.0        # temperature
  sigma_v: 0.30      # control noise std (v) [m/s]
  sigma_omega: 0.50  # control noise std (omega) [rad/s]
  seed: 12345

# === robot params ===
robot:
  wheel_radius: 0.10
  track_width: 0.40
  v_max: 2.0
  omega_max: 2.0
  a_max: 2.0
  alpha_max: 3.0
  robot_radius: 0.32

# === slip (M1: identity; M3: aktif) ===
slip:
  kappa_v: 1.0
  kappa_w: 1.0
  v_y: 0.0

# === cost weights (başlangıç; M4'te tune edilir) ===
cost:
  w_lat: 50.0
  w_head: 10.0
  w_prog: 5.0
  w_du: 1.0
  w_coll_hard: 1.0e6
  w_coll_soft: 100.0
  w_slope: 20.0
  w_rough: 10.0
  w_trav: 30.0
  w_rollover: 1.0e5
  k_accel: 5.0        # accel-limit soft penalty iç katsayısı
  safe_hard: 0.10     # [m] hard barrier eşiği (clearance)
  safe_soft: 0.50     # [m] soft margin başlangıcı
  rollover_slope: 0.45  # [rad] (~26°) rollover eşiği

# === maps ===
maps:
  esdf:      { res: 0.05, nx: 400, ny: 400, origin_x: -2.0, origin_y: -10.0 }
  elevation: { res: 0.10, nx: 200, ny: 200, origin_x: -2.0, origin_y: -10.0 }

# === reference ===
reference:
  ds: 0.05            # resample arc-length [m]
  window_N: 64        # device nearest-search penceresi

# === sim ===
sim:
  timestep: 0.002
  pos_noise_std: 0.0  # GPS/RTK proxy gürültüsü (0 = ground-truth)
```

---

## 9. Milestone'lar (artımlı — her milestone çalışan bir sistem bırakır)

### M0 — Toolchain + sahne + boş köprü
**Hedef:** Phase 0 tamam; CUDA/pybind11 derleniyor; MuJoCo sahnesi açılıyor ve robot manuel sürülüyor;
Python'dan CUDA stub çağrısı çalışıyor.

- [ ] Phase 0 (Bölüm 1) tüm kutuları işaretli, `docs/PHASE0_REPORT.md` yazıldı.
- [ ] `CMakeLists.txt` + `pyproject.toml` ile `cumpc_core` modülü derleniyor (`uv pip install -e .`).
- [ ] `bindings/pybind_module.cpp`'te **stub** `MPPIController.step()` (sabit `[0,0]` döndürür) Python'dan
      import edilip çağrılabiliyor: `import cumpc_core` çalışıyor.
- [ ] `device_buffer.cuh` + `cuda_check.cuh` mevcut; basit bir `DeviceBuffer<float>` upload/download testi geçiyor.
- [ ] `mujoco_harness/scene.xml` (Phase 0 robot kararı uygulanmış) `mujoco.viewer` ile açılıyor.
- [ ] `examples/drive_manual.py`: scripted/klavye ile robot ileri/dönüş yapıyor (aktüatör dönüşümü çalışıyor).
- [ ] Sensör okuma (`sensors.py`): `framepos+framequat → [px,py,theta]` doğru; konsola basılıyor.

**DoD:** `import cumpc_core` + manuel sürüş + stub `step` çağrısı uçtan uca çalışıyor; derleme uyarısız.

---

### M1 — Gerçek CUDA MPPI core + exact-arc dynamics + cm-takip (engel/terrain YOK)
**Hedef:** Sampling + rollout + reduction kernelleri tam; diff-drive exact-arc dynamics; düz/eğri/S-curve
path'te cm hassasiyetli takip. `compute-sanitizer` temiz.

- [ ] `dynamics/diff_drive.cuh`: `diff_drive_step` EXACT ARC; `test_dynamics.py` analitik referansa karşı
      geçiyor (düz, sabit-yarıçap çember; hata < 1e-5).
- [ ] `sampling.cu`: cuRAND ile `U[k,t]` üretimi + clamp; `test_sampling.py` (ortalama≈U_nom, std≈sigma,
      sabit seed determinizm) geçiyor.
- [ ] `rollout.cu`: thread-per-rollout, tracking cost (`w_lat`+`w_head`+`w_prog`+`w_du`+accel) ile `S[k]`.
      (Bu milestone'da obstacle/terrain cost **devre dışı**, slip = identity.)
- [ ] `reduction.cu`: min (`rho`), `weights`, `eta`, `weighted_update`; `test_mppi.py` smoke geçiyor.
- [ ] `mppi.cu`: `step()` orkestrasyonu + warm-start (shift) çalışıyor.
- [ ] Harness `track_straight.py` / `track_curve.py` / `track_scurve.py` ile robot path'i takip ediyor.
- [ ] **cm-takip:** ground-truth framepos'a karşı cross-track **RMS < 0.03 m** (düz + eğri); S-curve'de
      **RMS < 0.05 m** (eşikler config-tunable; M4'te iyileştirilir).
- [ ] `compute-sanitizer --tool memcheck` (ve `--tool racecheck`) **temiz** (0 hata).
- [ ] MPPI loop-rate ölçülüyor ve raporlanıyor (hedef: kontrol periyoduyla uyumlu, ≥ 20 Hz).

**DoD:** Saf CUDA MPPI, MuJoCo'da üç path tipinde cm-takip; sanitizer temiz; loop-rate metriği var.

---

### M2 — Obstacle costmap/ESDF (device array) + collision cost
**Hedef:** ESDF device array'i; rollout kernelinde **production-identical** `esdf.query` ile collision cost
(hard barrier + soft margin). Robot engellerden kaçarken path'i takip ediyor.

- [ ] `perception/costmap.cpp`: privileged geom konumlarından ESDF grid (host) + `set_esdf` ile D2H upload.
- [ ] `map_view.cuh`: `MapView::query` bilinear; `test_query.py` (bilinear doğruluk, sınır clamp) geçiyor.
- [ ] `rollout.cu` cost'a obstacle terimi eklendi: `d_clear < safe_hard → w_coll_hard`,
      `d_clear < safe_soft → w_coll_soft·(margin)²`.
- [ ] `examples/full_course.py` (engelli, terrain düz): robot engellere çarpmadan path'i takip ediyor.
- [ ] **Collision sayısı = 0** (geom mesafesi < `robot_radius` olayları), cross-track RMS makul kalıyor.
- [ ] `compute-sanitizer` temiz (map okumaları dahil; OOB yok).

**DoD:** Engelli kursta sıfır çarpışma + path takibi; ESDF query production ile aynı.

---

### M3 — Terrain features (device array) + terrain cost + slip
**Hedef:** Elevation device array'i (slope/roughness/traversability); terrain cost + slip-aware rollout
(`kappa_v`/`kappa_w`/`v_y`). Eğimli/pürüzlü zeminde slip compensation otantik test ediliyor.

- [ ] `perception/elevation.cpp`: `hfield`'dan `slope` (gradient), `roughness` (lokal varyans),
      `traversability` (slope+rough fonksiyonu) → grid + `set_elevation` upload.
- [ ] `map_view.cuh`: `query_terrain` bilinear → `TerrainCell`.
- [ ] `rollout.cu` cost'a terrain terimleri: `w_slope`, `w_rough`, `w_trav`, `w_rollover` (slope eşiği).
- [ ] Slip aktif: `SlipParams` terrain query'sinden beslenir (ör. düşük trav → `kappa_v`↓, `v_y`↑) **veya**
      config sabit estimate. `diff_drive_step` slip kolu çalışıyor.
- [ ] Eğimli/pürüzlü zeminde robot: rollover guard sayesinde dik eğimlerden kaçınıyor; düşük-trav bölgeleri
      tercih etmiyor; path'i takip ediyor.
- [ ] **Slip karşılaştırması:** slip-naive (kappa=1, v_y=0) vs slip-aware koşu → slip-aware'de cross-track
      RMS daha düşük (RESULTS.md'de tablo).
- [ ] `compute-sanitizer` temiz.

**DoD:** Terrain + slip aktif; rollover/trav davranışı gözlemleniyor; slip-aware vs slip-naive farkı ölçülü.

---

### M4 — Metrikler + tuning + profiling + sonuç
**Hedef:** Tam metrik seti; cost ağırlığı tuning; `nsys`/`ncu` profili; sonuçların raporlanması.

- [ ] `metrics/evaluate.py`: cross-track RMS, max cross-track, collision sayısı, loop-rate (Hz),
      tamamlanma süresi → otomatik rapor.
- [ ] `metrics/plot_results.py`: trajeler (ref vs gerçek), hata zaman serisi, hız profili grafiklerini üretiyor.
- [ ] **Tuning:** cost ağırlıkları (özellikle `w_lat`/`w_head`/`w_prog` ve `lambda`, `sigma_*`) üç path +
      engel + terrain için ayarlandı; final değerler config'e işlendi.
- [ ] `nsys profile`: MPPI loop timeline; kernel başına süre + bottleneck tespiti `docs/RESULTS.md`'de.
- [ ] `ncu`: en pahalı kernel(ler) (büyük olasılıkla `rollout_cost_kernel`) için occupancy, memory throughput,
      warp efficiency raporu; en az bir somut optimizasyon notu (ör. layout/coalescing).
- [ ] **Final hedefler:** düz/eğri RMS < 0.03 m, S-curve < 0.05 m, collision = 0, loop-rate ≥ 20 Hz (RTX 4070).
- [ ] `docs/RESULTS.md` tüm metrikleri + profiling özetini + slip-aware/naive karşılaştırmasını içeriyor.

**DoD:** Sayısal sonuçlar + profil raporu eksiksiz; hedef metrikler karşılanmış; config final.

---

## 10. Build / Çalıştırma Komutları

### Ortam (uv)
```bash
uv venv && source .venv/bin/activate
uv python pin 3.10
uv pip install mujoco numpy pyyaml matplotlib   # PyTorch YOK
uv pip install pybind11 scikit-build-core        # build deps (veya Phase 0 fallback)
```

### CUDA core + pybind11 modül build
**Yol A — scikit-build-core (önerilen, tek komut):**
```bash
uv pip install -e .            # CMake'i tetikler, cumpc_core.*.so paketi içine kurulur
```
**Yol B — manuel CMake fallback:**
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j
# çıkan cumpc_core*.so import yoluna kopyalanır / PYTHONPATH'e eklenir
```

### Çalıştırma
```bash
python examples/drive_manual.py                 # M0
python examples/track_straight.py               # M1
python examples/track_curve.py                  # M1
python examples/track_scurve.py                 # M1 (+ M2/M3 bayraklarıyla engel/terrain)
python examples/full_course.py                  # M2 + M3
python metrics/evaluate.py --run full_course     # M4 metrik
python metrics/plot_results.py --run full_course # M4 grafik
```

### Profiling / doğruluk
```bash
# Bellek/yarış doğruluğu (M1+ her milestone'da temiz olmalı)
compute-sanitizer --tool memcheck  python examples/track_scurve.py
compute-sanitizer --tool racecheck python examples/track_scurve.py
compute-sanitizer --tool initcheck python examples/track_scurve.py
compute-sanitizer --tool synccheck python examples/track_scurve.py

# Sistem timeline (M4)
nsys profile -o docs/nsys_report --force-overwrite true python examples/full_course.py

# Kernel-detay (M4) — rollout kernelini hedefle
ncu --set full -o docs/ncu_report -k rollout_cost_kernel --launch-count 5 python examples/full_course.py
```

---

## 11. Metrik / Validasyon (MuJoCo içinde)

| Metrik | Tanım | Hedef |
|---|---|---|
| **Cross-track RMS** | ground-truth `framepos` vs reference path dik mesafe, RMS | < 0.03 m (düz/eğri), < 0.05 m (S-curve) |
| **Max cross-track** | en kötü dik sapma | raporlanır |
| **Collision sayısı** | geom mesafesi < `robot_radius` olay sayısı | 0 |
| **MPPI loop-rate** | `step()` süresinden Hz | ≥ 20 Hz (RTX 4070) |
| **Kernel profili** | `rollout_cost_kernel` occupancy/throughput | `ncu` raporu + ≥1 optimizasyon notu |
| **Slip karşılaştırma** | slip-aware vs slip-naive RMS | slip-aware daha düşük |

- Ground-truth daima MuJoCo `framepos`'tan (RTK proxy). Gürültü çalışmaları için `pos_noise_std` artırılabilir.
- Tüm koşular `metrics/evaluate.py` ile tekrarlanabilir (sabit seed) ve `docs/RESULTS.md`'ye yazılır.

---

## 12. Test (`test/`)
- `test_dynamics.py` — exact-arc: düz çizgi, sabit-yarıçap çember, omega→0 limiti; slip identity vs aktif.
- `test_sampling.py` — noise istatistiği (mean/std), clamp sınırları, sabit-seed determinizm.
- `test_query.py` — bilinear interpolation doğruluğu (bilinen grid), sınır clamp, ESDF işaret doğruluğu.
- `test_mppi.py` — tek-adım `step()` boyut/sınır kontrolü, warm-start shift doğruluğu, control limit aşımı yok.

---

## 13. CLAUDE.md için invariant'lar (uygulama sırasında ihlal edilmez)

- [ ] MPPI çekirdeği **saf custom CUDA**'dır; PyTorch eklenmez.
- [ ] Tüm kontrol matematiği CUDA tarafında; Python harness yalnız sensör I/O + aktüatör + viz.
- [ ] **Naming (Bölüm 5)** birebir korunur — production cuMPC'ye taşıma garantisi.
- [ ] Integration **EXACT ARC** (Euler değil).
- [ ] Map **query** rollout kernelinde production-identical (build basit olsa da query değişmez).
- [ ] Tüm device buffer'lar RAII (`DeviceBuffer`); her CUDA çağrısı `CUDA_CHECK`.
- [ ] Her milestone sonunda `compute-sanitizer` temiz (M1+).
- [ ] Her milestone artımlıdır: bitince çalışan bir sistem bırakır; bir sonraki üstüne ekler.

---

## 14. Kapsam Dışı (bilinçli)
GP / ensemble / learned residual, CVaR, evidential DL, active perception, ROS2 entegrasyonu, gerçek donanım.
Bunlar production cuMPC kabuğunda ele alınır; bu spec yalnız **gerçek CUDA çekirdeğin MuJoCo'da geliştirilmesi**
ile sınırlıdır.
