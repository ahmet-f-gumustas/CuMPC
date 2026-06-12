#pragma once
#include <curand_kernel.h>

// host launcher'lar — kernel'ler sampling.cu içinde
void init_rng(curandState* rng, unsigned long long seed, int K);
void sample_controls(const float* U_nom, float* U, curandState* rng,
                     float sigma_v, float sigma_omega,
                     float v_max, float omega_max, int K, int H);
