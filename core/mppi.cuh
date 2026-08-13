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
// Bir config yayınının kimliği. Çağıran runtime, uyguladığı epoch ile planner'ın GERÇEKTEN okuduğu
// epoch'u karşılaştırabilsin diye monoton artar; eşit değillerse geçiş uygulanmamıştır ve bunu
// varsaymak yerine görmek gerekir.
using ConfigEpoch = unsigned long long;

class MPPI {
public:
    explicit MPPI(const MPPIConfig& cfg);
    ~MPPI();

    MPPI(const MPPI&) = delete;
    MPPI& operator=(const MPPI&) = delete;

    // x0=[px,py,theta]; ref_host=[N*3] (x,y,heading); → [v,omega]
    //
    // BLOKLAYAN, geriye dönük uyum için: enqueue() + tamamlanmayı bekle + result(). Mevcut
    // tüketiciler (pybind modülü, cumpc-mujoco harness'ı) bu imzayı kullanmaya devam eder.
    float2 step(float3 x0, const float* ref_host, int N);

    // ---- BLOKLAMAYAN yol (BudgetRT runtime sözleşmesi) --------------------------------
    //
    // Ölçüm (BudgetRT `tests/perf/stage_d_task5_planner_costs/`): step()'in %77'si, 8 baytlık
    // control çıktısının senkron D2H'sinde geçiyor - kopya yavaş olduğu için değil, host'un orada
    // durup kendisinden önce kuyruğa girmiş her kernel'i beklediği için. enqueue/poll ayrımı GPU'yu
    // hızlandırmaz; o süreyi host'a geri verir.
    //
    // enqueue(): işi stream'e koyar, tamamlanma event'ini kaydeder ve DÖNER. Beklemez.
    void enqueue(float3 x0, const float* ref_host, int N);
    // poll(): tamamlanma event'ini SORGULAR (cudaEventQuery), beklemez. true = sonuç hazır.
    bool poll() const;
    // wait(): tamamlanmayı bekler. Kritik yolda çağrılmaz; step()'in ve testlerin kullandığı yol.
    void wait() const;
    // result(): son tamamlanan enqueue'nun kontrolü. poll() true dönmeden okunması tanımsızdır.
    float2 result() const;

    void set_esdf(const float* host, MapMeta meta);
    // K9(b): maliyet gridi ZATEN CİHAZDA olduğunda kopyalamadan benimser. BudgetRT'nin overlay'i
    // çıktısını cihazda üretir; onu host'a indirip geri yüklemek ölçülen ~208 µs/tick'lik senkron
    // H2D'nin tamamıdır. `device` işaretçisinin sahibi ÇAĞIRANDIR ve en az bir sonraki enqueue()
    // tamamlanana kadar geçerli kalmalıdır.
    void set_cost_grid_device(const float* device, MapMeta meta);
    void set_elevation(const float* host, MapMeta meta);
    void set_slip(const SlipParams& slip) { active_().slip = slip; }

    // ---- İKİ TAMPONLU IMMUTABLE CONFIG (BudgetRT §18.3'ün adapter'dan istediği şey) --------
    //
    // Aktif config, bir tick koşarken DEĞİŞTİRİLEMEZ. Değişiklik iki adımdır ve ikisi ayrı yerde
    // durur:
    //
    //   prepare(cfg) — doğrular ve PASİF slot'a hazırlar. Pahalı iş (K/H büyüdüyse tampon
    //                  büyütmesi) burada, yani tick yolunun DIŞINDA yapılır. Aktif config'e
    //                  dokunmaz; başarısız olursa hiçbir şey değişmemiştir.
    //   commit()     — slot'u takas eder ve epoch'u artırır. CUDA çağrısı yoktur; bu yüzden bir
    //                  frame sınırında yapılabilir. Uçuşta iş varken REDDEDİLİR.
    //
    // K ve H'nin sınıf (ii) olmasının sebebi tam olarak budur: değişmeleri yeniden ayırma
    // gerektirir, o da prepare()'e aittir. Sınıf (i) yapmak (kapasite-maksimum tahsis) ayrı bir
    // karardır ve bilerek ertelenmiştir.
    [[nodiscard]] bool prepare(const MPPIConfig& cfg);
    [[nodiscard]] bool has_prepared() const { return has_prepared_; }
    void abort_prepared() { has_prepared_ = false; }
    // Takas edilen epoch'u döner; reddedilirse aktif epoch'u değiştirmeden döner.
    ConfigEpoch commit();

    ConfigEpoch active_epoch() const { return epoch_; }
    // SON enqueue()'nun gerçekten okuduğu epoch. commit() sonrası bir tick koşmadan bu değer
    // değişmez - "yayınlandı" ile "uygulandı" iki ayrı olgudur ve karıştırılmaları, hiç uygulanmamış
    // bir geçişin uygulanmış sayılmasıdır.
    ConfigEpoch observed_epoch() const { return observed_epoch_; }
    void reset();  // warm-start + rng + u_prev sıfırla (tam determinizm)

    void nominal(float* out_h2) const;        // [H*2]
    void last_rollouts(float* out_kh3) const; // [K*H*3]
    const MPPIConfig& config() const { return active_(); }
    MapView esdf_view() const;
    MapView elev_view() const;

    // Hangi stream'de çalıştığı, çağıranın kendi bağımlılıklarını kurabilmesi için görünür
    // (cudaStreamWaitEvent). Sahibi bu sınıftır ve ömrü nesnenin ömrüdür.
    cudaStream_t stream() const { return stream_; }

private:
    void submit_(float3 x0, const float* ref_host, int N);

    MPPIConfig& active_() { return cfg_[active_slot_]; }
    const MPPIConfig& active_() const { return cfg_[active_slot_]; }
    [[nodiscard]] bool grow_for_(const MPPIConfig& cfg);

    // İKİ SLOT. Aktif olan okunur, diğeri hazırlanır; commit yalnız indeksi çevirir.
    MPPIConfig cfg_[2];
    int active_slot_ = 0;
    bool has_prepared_ = false;
    ConfigEpoch epoch_ = 1;
    ConfigEpoch observed_epoch_ = 0;

    // NON-BLOCKING olarak yaratılır: legacy default stream ile ÖRTÜK senkronizasyon kurmaz. Bu,
    // çağıran runtime'ın kanal ayrımının (BudgetRT §6.3 k4) planner tarafından bozulmamasının tek
    // yoludur.
    cudaStream_t stream_ = nullptr;
    // Tamamlanma event'i (timing kapalı: sorgulanır, ölçülmez).
    cudaEvent_t done_ = nullptr;
    // PINNED host çıktısı [2]. Pageable olsaydı D2H yine senkronlaşırdı ve ayrımın anlamı kalmazdı.
    float* out_pinned_ = nullptr;

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
    // K9(b): çağıranın sahibi olduğu cihaz işaretçisi. nullptr ise kendi tamponu kullanılır.
    const float* esdf_external_ = nullptr;
    MapMeta esdf_meta_{};
    MapMeta elev_meta_{};
    bool has_esdf_ = false;
    bool has_elev_ = false;

    int ref_cap_ = 0;
    float2 u_prev_{0.0f, 0.0f};
};
