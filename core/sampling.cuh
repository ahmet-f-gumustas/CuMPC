#pragma once
#include <curand_kernel.h>

// host launcher'lar — kernel'ler sampling.cu içinde
// `stream` is REQUIRED on both - see the note in rollout.cuh for why it is not defaulted.
void init_rng(curandState* rng, unsigned long long seed, int K, cudaStream_t stream);
void sample_controls(const float* U_nom, float* U, curandState* rng,
                     float sigma_v, float sigma_omega,
                     float v_max, float omega_max, int K, int H, cudaStream_t stream);
