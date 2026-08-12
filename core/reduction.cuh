#pragma once
#include <cuda_runtime.h>


// host launcher'lar — kernel'ler reduction.cu içinde.
// Not (spec §6.2'den bilinçli sapma): rho ve eta device scalar pointer olarak
// taşınır — host'a indirip değerle geçmek her step'te gereksiz D2H sync olurdu.
// eta_partial: [num_blocks(K)] scratch; eta: [1].
// `stream` is REQUIRED on all three - see the note in rollout.cuh for why it is not defaulted.
int reduction_num_blocks(int K);
void reduce_min(const float* S, float* rho, int K, cudaStream_t stream);  // rho = min_k S[k]
void compute_weights(const float* S, const float* rho, float lambda,
                     float* wgt, float* eta_partial, float* eta, int K, cudaStream_t stream);
void weighted_update(const float* U, const float* wgt, const float* eta,
                     float* U_nom, int K, int H, cudaStream_t stream);
