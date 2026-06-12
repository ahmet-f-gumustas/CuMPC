#pragma once
#include "core/device_buffer.cuh"
#include "core/types.cuh"
#include "perception/map_view.cuh"

struct MPPIConfig {
    int K = 2048;                     // num_samples
    int H = 40;                       // horizon
    float dt = 0.05f;                 // rollout timestep [s]
    float lambda = 1.0f;              // temperature
    float sigma_v = 0.30f;            // control noise std (v) [m/s]
    float sigma_omega = 0.50f;        // control noise std (omega) [rad/s]
    unsigned long long seed = 12345;  // cuRAND seed (determinizm)
    RobotParams robot{};
    SlipParams slip{};
    CostWeights weights{};
    CostParams cost{};
};

// MPPI orkestrasyonu: sample → rollout → min → weights → update → warm-start.
// Tüm device belleği RAII (DeviceBuffer); harness yalnızca step() çağırır.
class MPPI {
public:
    explicit MPPI(const MPPIConfig& cfg);

    // x0=[px,py,theta]; ref_host=[N*3] (x,y,heading); → [v,omega]
    float2 step(float3 x0, const float* ref_host, int N);

    void set_esdf(const float* host, MapMeta meta);
    void set_elevation(const float* host, MapMeta meta);
    void set_slip(const SlipParams& slip) { cfg_.slip = slip; }
    void reset();  // warm-start + rng + u_prev sıfırla (tam determinizm)

    void nominal(float* out_h2) const;        // [H*2]
    void last_rollouts(float* out_kh3) const; // [K*H*3]
    const MPPIConfig& config() const { return cfg_; }
    MapView esdf_view() const;
    MapView elev_view() const;

private:
    MPPIConfig cfg_;

    DeviceBuffer<float> U_;            // [K*H*2] örneklenen kontroller
    DeviceBuffer<float> U_nom_;        // [H*2] nominal (warm-start)
    DeviceBuffer<float> shift_;        // [H*2] warm-start shift scratch
    DeviceBuffer<float> S_;            // [K] rollout cost
    DeviceBuffer<float> traj_;         // [K*H*3] debug/viz
    DeviceBuffer<float> wgt_;          // [K]
    DeviceBuffer<float> eta_partial_;  // [num_blocks]
    DeviceBuffer<float> eta_;          // [1]
    DeviceBuffer<float> rho_;          // [1]
    DeviceBuffer<float> ref_;          // [cap*3] reference window
    // curandState[K] — opak byte buffer: header'ı curand_kernel.h'den bağımsız
    // tutar (pybind TU'su host-only g++ ile derlenir)
    DeviceBuffer<unsigned char> rng_;

    DeviceBuffer<float> esdf_data_;    // [ny*nx]
    DeviceBuffer<float> elev_data_;    // [ny*nx*3]
    MapMeta esdf_meta_{};
    MapMeta elev_meta_{};
    bool has_esdf_ = false;
    bool has_elev_ = false;

    int ref_cap_ = 0;
    float2 u_prev_{0.0f, 0.0f};
};
