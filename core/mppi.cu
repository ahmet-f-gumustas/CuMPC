#include <curand_kernel.h>

#include <stdexcept>

#include "core/cuda_check.cuh"
#include "core/mppi.cuh"
#include "core/reduction.cuh"
#include "core/rollout.cuh"
#include "core/sampling.cuh"

MPPI::MPPI(const MPPIConfig& cfg) : cfg_(cfg)
{
    if (cfg_.K <= 0 || cfg_.H <= 1) throw std::invalid_argument("MPPIConfig: K > 0, H > 1 gerekli");
    const int K = cfg_.K, H = cfg_.H;
    U_.reset((size_t)K * H * 2);
    U_nom_.reset((size_t)H * 2);
    shift_.reset((size_t)H * 2);
    S_.reset(K);
    traj_.reset((size_t)K * H * 3);
    wgt_.reset(K);
    eta_partial_.reset(reduction_num_blocks(K));
    eta_.reset(1);
    rho_.reset(1);
    rng_.reset((size_t)K * sizeof(curandState));
    reset();
}

void MPPI::reset()
{
    U_nom_.zero();
    u_prev_ = {0.0f, 0.0f};
    init_rng(reinterpret_cast<curandState*>(rng_.get()), cfg_.seed, cfg_.K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void MPPI::set_esdf(const float* host, MapMeta meta)
{
    esdf_data_.reset((size_t)meta.nx * meta.ny);
    esdf_data_.upload(host, (size_t)meta.nx * meta.ny);
    esdf_meta_ = meta;
    has_esdf_ = true;
}

void MPPI::set_elevation(const float* host, MapMeta meta)
{
    elev_data_.reset((size_t)meta.nx * meta.ny * 3);
    elev_data_.upload(host, (size_t)meta.nx * meta.ny * 3);
    elev_meta_ = meta;
    has_elev_ = true;
}

MapView MPPI::esdf_view() const
{
    return {has_esdf_ ? esdf_data_.get() : nullptr, esdf_meta_};
}

MapView MPPI::elev_view() const
{
    return {has_elev_ ? elev_data_.get() : nullptr, elev_meta_};
}

float2 MPPI::step(float3 x0, const float* ref_host, int N)
{
    if (N < 2) throw std::invalid_argument("reference window: N >= 2 gerekli");
    const int K = cfg_.K, H = cfg_.H;

    // reference window H2D
    if (N > ref_cap_) {
        ref_.reset((size_t)N * 3);
        ref_cap_ = N;
    }
    ref_.upload(ref_host, (size_t)N * 3);
    const RefWindow ref{ref_.get(), N};

    // 1) SAMPLE: U = U_nom + eps, clamp
    sample_controls(U_nom_.get(), U_.get(), reinterpret_cast<curandState*>(rng_.get()),
                    cfg_.sigma_v, cfg_.sigma_omega,
                    cfg_.robot.v_max, cfg_.robot.omega_max, K, H);

    // 2) ROLLOUT: S[k] = Σ_t step_cost
    rollout_cost(U_.get(), x0, u_prev_, S_.get(), traj_.get(),
                 cfg_.robot, cfg_.slip, cfg_.weights, cfg_.cost,
                 ref, esdf_view(), elev_view(), cfg_.dt, K, H);

    // 3) BASELINE: rho = min_k S[k]
    reduce_min(S_.get(), rho_.get(), K);

    // 4) WEIGHTS: wgt = exp(-(1/λ)(S-ρ)), eta = Σ wgt
    compute_weights(S_.get(), rho_.get(), cfg_.lambda,
                    wgt_.get(), eta_partial_.get(), eta_.get(), K);

    // 5) UPDATE: U_nom = (1/eta) Σ_k wgt[k]·U[k,·]
    weighted_update(U_.get(), wgt_.get(), eta_.get(), U_nom_.get(), K, H);
    CUDA_CHECK(cudaGetLastError());

    // 6) OUTPUT: control = U_nom[0]  (D2H, implicit sync)
    float out[2];
    U_nom_.download(out, 2);

    // 7) WARMSTART: U_nom[0..H-2] ← U_nom[1..H-1]; U_nom[H-1] ← U_nom[H-2]
    CUDA_CHECK(cudaMemcpy(shift_.get(), U_nom_.get() + 2,
                          (size_t)(H - 1) * 2 * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(U_nom_.get(), shift_.get(),
                          (size_t)(H - 1) * 2 * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(U_nom_.get() + (size_t)(H - 1) * 2, U_nom_.get() + (size_t)(H - 2) * 2,
                          2 * sizeof(float), cudaMemcpyDeviceToDevice));

    u_prev_ = {out[0], out[1]};
    return {out[0], out[1]};
}

void MPPI::nominal(float* out_h2) const
{
    U_nom_.download(out_h2, (size_t)cfg_.H * 2);
}

void MPPI::last_rollouts(float* out_kh3) const
{
    traj_.download(out_kh3, (size_t)cfg_.K * cfg_.H * 3);
}
