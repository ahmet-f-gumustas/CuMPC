#pragma once
#include <cuda_runtime.h>

#include "core/types.cuh"
#include "perception/map_view.cuh"

// host launcher — rollout_cost_kernel rollout.cu içinde.
// traj: [K*H*3] (px,py,theta) debug/viz çıktısı (her zaman yazılır, ucuz).
// u_prev: bir önceki step'te uygulanan control (t=0 için Δu/accel referansı).
// `stream` is REQUIRED and is not defaulted. A default would let a caller launch on the legacy
// default stream by omission, and the legacy default stream synchronises implicitly with every other
// stream in the process - which is exactly the serialisation this signature exists to remove.
void rollout_cost(const float* U, float3 x0, float2 u_prev, float* S, float* traj,
                  RobotParams rp, SlipParams slip, CostWeights w, CostParams cp,
                  RefWindow ref, MapView esdf, MapView elev,
                  float dt, int K, int H, cudaStream_t stream);
