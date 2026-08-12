#include "core/rollout.cuh"
#include "dynamics/diff_drive.cuh"

namespace {
// 64'lük blok: K=2048 thread-per-rollout → 32 blok, 36 SM'e daha iyi dağılır
// (128'e karşı ~%2 kazanç; asıl sınır thread başına seri iş — bkz. RESULTS.md)
constexpr int kBlock = 64;
}

// accel-limit soft penalty: limit aşımının karesi × k_accel
__device__ __forceinline__
float accel_pen(float a, float a_max, float k_accel)
{
    const float ex = fabsf(a) - a_max;
    return ex > 0.0f ? k_accel * ex * ex : 0.0f;
}

// Window içinde en yakın referans noktasının index'i (windowed-nearest, spec §6.7).
__device__ __forceinline__
int nearest_ref_idx(const RefWindow& ref, float px, float py)
{
    int best = 0;
    float best_d2 = 3.4e38f;
    for (int i = 0; i < ref.N; ++i) {
        const float dx = px - ref.pts[i * 3 + 0];
        const float dy = py - ref.pts[i * 3 + 1];
        const float d2 = dx * dx + dy * dy;
        if (d2 < best_d2) { best_d2 = d2; best = i; }
    }
    return best;
}

__device__ __forceinline__
RefPoint ref_at(const RefWindow& ref, int i)
{
    return {ref.pts[i * 3 + 0], ref.pts[i * 3 + 1], ref.pts[i * 3 + 2]};
}

// Tek rollout adımının cost'u (spec §6.4 referans kontratı).
__device__ __forceinline__
float step_cost(float px, float py, float theta,
                float v, float omega, float v_prev, float omega_prev,
                const RefPoint& ref, const CostWeights& w, const CostParams& cp,
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
    c += w.w_du * (dv * dv + dw * dw);

    // 5) accel limit (soft) — control limiti sampling'de hard-clamp edilir
    c += accel_pen(dv / dt, rp.a_max, cp.k_accel);
    c += accel_pen(dw / dt, rp.alpha_max, cp.k_accel);

    // 6) obstacle — ESDF query (PRODUCTION-IDENTICAL); map yoksa terim devre dışı
    if (esdf.valid()) {
        const float d = esdf.query(px, py);  // signed distance, engele [m]
        const float d_clear = d - rp.robot_radius;
        if (d_clear < cp.safe_hard) c += w.w_coll_hard;  // hard barrier
        if (d_clear < cp.safe_soft) {
            const float m = cp.safe_soft - d_clear;
            c += w.w_coll_soft * m * m;                  // soft margin
        }
    }

    // 7) terrain — elevation query (PRODUCTION-IDENTICAL); map yoksa devre dışı
    if (elev.valid()) {
        const TerrainCell t = elev.query_terrain(px, py);
        c += w.w_slope * t.slope * t.slope;
        c += w.w_rough * t.roughness;
        c += w.w_trav  * (1.0f - t.traversability);
        if (t.slope > cp.rollover_slope) c += w.w_rollover;  // rollover guard
    }

    return c;
}

// thread-per-rollout: 1 thread = 1 tam rollout (H adım dynamics + cost akümülasyonu).
// State registers'da; global okuma yalnız U[k,·] + ref window + map query (read-only).
__global__ void rollout_cost_kernel(
    const float* __restrict__ U,     // [K*H*2]
    const float3              x0,    // [px,py,theta]
    const float2              u_prev,
    float*       __restrict__ S,     // [K]  (out: toplam cost)
    float*       __restrict__ traj,  // [K*H*3] (out: debug/viz)
    RobotParams  rp, SlipParams slip, CostWeights w, CostParams cp,
    RefWindow    ref,
    MapView      esdf, MapView elev,
    float dt, int K, int H)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;

    float px = x0.x, py = x0.y, theta = x0.z;
    float v_prev = u_prev.x, omega_prev = u_prev.y;
    float cost = 0.0f;

    for (int t = 0; t < H; ++t) {
        const float v     = U[(k * H + t) * 2 + 0];
        const float omega = U[(k * H + t) * 2 + 1];

        diff_drive_step(px, py, theta, v, omega, slip, dt);

        const RefPoint rpt = ref_at(ref, nearest_ref_idx(ref, px, py));
        cost += step_cost(px, py, theta, v, omega, v_prev, omega_prev,
                          rpt, w, cp, rp, esdf, elev, dt);

        traj[(k * H + t) * 3 + 0] = px;
        traj[(k * H + t) * 3 + 1] = py;
        traj[(k * H + t) * 3 + 2] = theta;

        v_prev = v;
        omega_prev = omega;
    }

    // terminal cost: lookahead'e PATH BOYUNCA (arc-length) kalan mesafe².
    // Öklidsel mesafe viraj kirişini ödüllendirip köşe kestirirdi; arc-length
    // yalnız gerçek path ilerlemesini sayar (anti-deadlock baskısı korunur).
    // ds, sabit arc-length resample'lı window'un ilk iki noktasından türetilir.
    const float dsx = ref.pts[3] - ref.pts[0];
    const float dsy = ref.pts[4] - ref.pts[1];
    const float ds = sqrtf(dsx * dsx + dsy * dsy);
    const int i_end = nearest_ref_idx(ref, px, py);
    const float d_term = (float)(ref.N - 1 - i_end) * ds;
    const float d_pen = fmaxf(d_term - cp.term_slack, 0.0f);
    cost += w.w_term * d_pen * d_pen;

    S[k] = cost;
}

void rollout_cost(const float* U, float3 x0, float2 u_prev, float* S, float* traj,
                  RobotParams rp, SlipParams slip, CostWeights w, CostParams cp,
                  RefWindow ref, MapView esdf, MapView elev,
                  float dt, int K, int H, cudaStream_t stream)
{
    const int grid = (K + kBlock - 1) / kBlock;
    rollout_cost_kernel<<<grid, kBlock, 0, stream>>>(U, x0, u_prev, S, traj,
                                          rp, slip, w, cp, ref, esdf, elev,
                                          dt, K, H);
}
