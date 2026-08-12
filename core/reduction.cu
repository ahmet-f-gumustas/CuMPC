#include <cfloat>

#include "core/reduction.cuh"

namespace {
constexpr int kBlock = 256;
}

int reduction_num_blocks(int K) { return (K + kBlock - 1) / kBlock; }

// float atomicMin — CAS döngüsü (min exact-associative olduğundan deterministik).
__device__ __forceinline__
void atomic_min_float(float* addr, float value)
{
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int;
    while (__int_as_float(old) > value) {
        const int assumed = old;
        old = atomicCAS(addr_as_int, assumed, __float_as_int(value));
        if (old == assumed) break;
    }
}

__global__ void set_inf_kernel(float* x) { *x = FLT_MAX; }

// blok-reduce (shared mem) + blok sonucu atomic min → rho
__global__ void reduce_min_kernel(const float* __restrict__ S,
                                  float* __restrict__ rho, int K)
{
    __shared__ float sh[kBlock];
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    sh[threadIdx.x] = (k < K) ? S[k] : FLT_MAX;
    __syncthreads();
    for (int s = kBlock / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] = fminf(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0) atomic_min_float(rho, sh[0]);
}

// wgt[k] = exp(-(1/lambda)(S[k]-rho)) ≤ 1 (rho baseline sayısal taşmayı önler)
// + blok-sum → eta_partial[blockIdx] (deterministik: atomicAdd YOK)
__global__ void weights_kernel(const float* __restrict__ S,
                               const float* __restrict__ rho, float lambda,
                               float* __restrict__ wgt,
                               float* __restrict__ eta_partial, int K)
{
    __shared__ float sh[kBlock];
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    float w = 0.0f;
    if (k < K) {
        w = expf(-(S[k] - *rho) / lambda);
        wgt[k] = w;
    }
    sh[threadIdx.x] = w;
    __syncthreads();
    for (int s = kBlock / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) eta_partial[blockIdx.x] = sh[0];
}

// eta = Σ partial — tek blok, sabit sıra → run-to-run deterministik
__global__ void finalize_eta_kernel(const float* __restrict__ eta_partial,
                                    float* __restrict__ eta, int num_partials)
{
    __shared__ float sh[kBlock];
    float acc = 0.0f;
    for (int i = threadIdx.x; i < num_partials; i += kBlock) acc += eta_partial[i];
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int s = kBlock / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) *eta = sh[0];
}

// U_nom[t,dim] = (1/eta) Σ_k wgt[k]·U[k,t,dim] — (t,dim) başına BİR BLOK,
// k üzerinde blok-paralel tree-reduce (sabit toplama sırası → deterministik).
// Not (M4 ncu bulgusu): thread-per-(t,dim) + k döngüsü versiyonu yalnız H*2=80
// thread kullanıp toplam GPU süresinin %40'ını alıyordu; blok-paralel sürüm
// K üzerindeki okumayı 256 thread'e dağıtır.
__global__ void weighted_update_kernel(const float* __restrict__ U,
                                       const float* __restrict__ wgt,
                                       const float* __restrict__ eta,
                                       float* __restrict__ U_nom, int K, int H)
{
    __shared__ float sh[kBlock];
    const int i = blockIdx.x;  // i = t*2 + dim, grid = H*2 blok
    if (i >= H * 2) return;
    float acc = 0.0f;
    for (int k = threadIdx.x; k < K; k += kBlock) acc += wgt[k] * U[k * H * 2 + i];
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int s = kBlock / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) U_nom[i] = sh[0] / *eta;
}

void reduce_min(const float* S, float* rho, int K, cudaStream_t stream)
{
    set_inf_kernel<<<1, 1, 0, stream>>>(rho);
    reduce_min_kernel<<<reduction_num_blocks(K), kBlock, 0, stream>>>(S, rho, K);
}

void compute_weights(const float* S, const float* rho, float lambda,
                     float* wgt, float* eta_partial, float* eta, int K, cudaStream_t stream)
{
    const int nb = reduction_num_blocks(K);
    weights_kernel<<<nb, kBlock, 0, stream>>>(S, rho, lambda, wgt, eta_partial, K);
    finalize_eta_kernel<<<1, kBlock, 0, stream>>>(eta_partial, eta, nb);
}

void weighted_update(const float* U, const float* wgt, const float* eta,
                     float* U_nom, int K, int H, cudaStream_t stream)
{
    weighted_update_kernel<<<H * 2, kBlock, 0, stream>>>(U, wgt, eta, U_nom, K, H);
}
