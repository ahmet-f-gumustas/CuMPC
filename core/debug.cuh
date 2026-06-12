#pragma once
#include <cuda_runtime.h>

#include "core/types.cuh"
#include "perception/map_view.cuh"

// Test/debug launcher'ları: production device fonksiyonlarını
// (diff_drive_step, MapView::query, query_terrain) birebir çağırır.
void debug_simulate(float3 x0, const float* U_dev, float* states_dev,
                    SlipParams slip, float dt, int H);
void debug_query(MapView map, const float* pts_dev, float* out_dev, int M);          // [M*2]→[M]
void debug_query_terrain(MapView map, const float* pts_dev, float* out_dev, int M);  // [M*2]→[M*3]
