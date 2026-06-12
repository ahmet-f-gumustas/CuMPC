#include <curand_kernel.h>

#include "core/sampling.cuh"

namespace {
constexpr int kBlock = 256;
}

// curandState dizisi bir kez init edilir (determinizm: seed config'ten,
// her thread kendi subsequence'i), her step ileri sürülür.
__global__ void init_rng_kernel(curandState* __restrict__ rng,
                                unsigned long long seed, int K)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    curand_init(seed, (unsigned long long)k, 0ull, &rng[k]);
}

// cuRAND ile K×H control örnekleme + hard clamp.
// U[k,t] = U_nom[t] + eps,  eps ~ N(0, diag(sigma_v, sigma_omega))
__global__ void sample_controls_kernel(
    const float* __restrict__ U_nom,   // [H*2]
    float*       __restrict__ U,       // [K*H*2]  (out)
    curandState* __restrict__ rng,     // [K]  (persistent, seed'li)
    float sigma_v, float sigma_omega,
    float v_max, float omega_max,
    int K, int H)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    curandState st = rng[k];  // local kopya: global state'e tek yazım
    for (int t = 0; t < H; ++t) {
        const float v = U_nom[t * 2 + 0] + curand_normal(&st) * sigma_v;
        const float w = U_nom[t * 2 + 1] + curand_normal(&st) * sigma_omega;
        U[(k * H + t) * 2 + 0] = fminf(fmaxf(v, -v_max), v_max);
        U[(k * H + t) * 2 + 1] = fminf(fmaxf(w, -omega_max), omega_max);
    }
    rng[k] = st;
}

void init_rng(curandState* rng, unsigned long long seed, int K)
{
    const int grid = (K + kBlock - 1) / kBlock;
    init_rng_kernel<<<grid, kBlock>>>(rng, seed, K);
}

void sample_controls(const float* U_nom, float* U, curandState* rng,
                     float sigma_v, float sigma_omega,
                     float v_max, float omega_max, int K, int H)
{
    const int grid = (K + kBlock - 1) / kBlock;
    sample_controls_kernel<<<grid, kBlock>>>(U_nom, U, rng, sigma_v, sigma_omega,
                                             v_max, omega_max, K, H);
}
