#include "core/debug.cuh"
#include "dynamics/diff_drive.cuh"

// Tek thread: verilen control dizisini deterministik ilerlet (test_dynamics için)
__global__ void simulate_kernel(float3 x0, const float* __restrict__ U,
                                float* __restrict__ states,
                                SlipParams slip, float dt, int H)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float px = x0.x, py = x0.y, theta = x0.z;
    for (int t = 0; t < H; ++t) {
        diff_drive_step(px, py, theta, U[t * 2 + 0], U[t * 2 + 1], slip, dt);
        states[t * 3 + 0] = px;
        states[t * 3 + 1] = py;
        states[t * 3 + 2] = theta;
    }
}

__global__ void query_kernel(MapView map, const float* __restrict__ pts,
                             float* __restrict__ out, int M)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;
    out[i] = map.query(pts[i * 2 + 0], pts[i * 2 + 1]);
}

__global__ void query_terrain_kernel(MapView map, const float* __restrict__ pts,
                                     float* __restrict__ out, int M)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;
    const TerrainCell t = map.query_terrain(pts[i * 2 + 0], pts[i * 2 + 1]);
    out[i * 3 + 0] = t.slope;
    out[i * 3 + 1] = t.roughness;
    out[i * 3 + 2] = t.traversability;
}

void debug_simulate(float3 x0, const float* U_dev, float* states_dev,
                    SlipParams slip, float dt, int H)
{
    simulate_kernel<<<1, 1>>>(x0, U_dev, states_dev, slip, dt, H);
}

void debug_query(MapView map, const float* pts_dev, float* out_dev, int M)
{
    query_kernel<<<(M + 255) / 256, 256>>>(map, pts_dev, out_dev, M);
}

void debug_query_terrain(MapView map, const float* pts_dev, float* out_dev, int M)
{
    query_terrain_kernel<<<(M + 255) / 256, 256>>>(map, pts_dev, out_dev, M);
}
