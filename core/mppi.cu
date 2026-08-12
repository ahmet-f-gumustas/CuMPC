#include <curand_kernel.h>

#include <stdexcept>

#include "core/cuda_check.cuh"
#include "core/mppi.cuh"
#include "core/reduction.cuh"
#include "core/rollout.cuh"
#include "core/sampling.cuh"

MPPI::MPPI(const MPPIConfig& cfg) : cfg_(cfg)
{
    if (cfg_.K <= 0 || cfg_.H <= 1) throw std::invalid_argument("MPPIConfig: K > 0, H > 1 gerekli");
    const int K = cfg_.K, H = cfg_.H;
    U_.reset((size_t)K * H * 2);
    U_nom_.reset((size_t)H * 2);
    shift_.reset((size_t)H * 2);
    S_.reset(K);
    traj_.reset((size_t)K * H * 3);
    wgt_.reset(K);
    eta_partial_.reset(reduction_num_blocks(K));
    eta_.reset(1);
    rho_.reset(1);
    rng_.reset((size_t)K * sizeof(curandState));

    // Kaynaklar BİR KEZ, burada. Sıcak yolda hiçbir şey yaratılmaz ve hiçbir şey serbest
    // bırakılmaz.
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&done_, cudaEventDisableTiming));
    CUDA_CHECK(cudaHostAlloc(&out_pinned_, 2 * sizeof(float), cudaHostAllocDefault));
    out_pinned_[0] = 0.0f;
    out_pinned_[1] = 0.0f;

    reset();
}

MPPI::~MPPI()
{
    // Yıkımda senkronizasyon YOK: outstanding iş varken cudaStreamDestroy/cudaEventDestroy tanımlı
    // davranıştır (kaynak, iş bittiğinde serbest kalır) ve burada beklemek, yıkımı GPU'ya bağımlı
    // kılardı. İşin bitmesini isteyen çağıran, yıkmadan önce wait() çağırır.
    if (out_pinned_) cudaFreeHost(out_pinned_);
    if (done_) cudaEventDestroy(done_);
    if (stream_) cudaStreamDestroy(stream_);
}

void MPPI::reset()
{
    // Ölçüldü: eski hâlde bir reset'in %97.5'i (106.6 µs'in 104.0 µs'i) buradaki
    // cudaDeviceSynchronize() idi - CİHAZ GENELİNDE bir bariyer, yani çağıran runtime'ın ayrı
    // tuttuğu bütün kanalları da durduran bir bekleme. Sıra zaten stream'in kendisinde kurulu:
    // init_rng bu stream'e girer ve ondan sonraki her iş arkasında sıralanır. Beklemeye gerek yok.
    U_nom_.zero_async(stream_);
    u_prev_ = {0.0f, 0.0f};
    init_rng(reinterpret_cast<curandState*>(rng_.get()), cfg_.seed, cfg_.K, stream_);
    CUDA_CHECK(cudaGetLastError());
}

void MPPI::set_esdf(const float* host, MapMeta meta)
{
    // resize() kapasite yeterliyse HİÇBİR CUDA çağrısı yapmaz; eski reset() burada tick başına bir
    // cudaFree + bir cudaMalloc yapıyordu (ölçüldü: ~103 µs/tick).
    const size_t cells = (size_t)meta.nx * meta.ny;
    esdf_data_.resize(cells);
    esdf_data_.upload_async(host, cells, stream_);
    esdf_meta_ = meta;
    has_esdf_ = true;
    esdf_external_ = nullptr;
}

void MPPI::set_cost_grid_device(const float* device, MapMeta meta)
{
    // K9(b). Kopya YOK: grid zaten cihazda. Sahiplik çağıranda kalır - bu sınıf onu ne serbest
    // bırakır ne de taşır, ve dolayısıyla ömrü çağıranın sözleşmesidir (bkz. header).
    esdf_external_ = device;
    esdf_meta_ = meta;
    has_esdf_ = (device != nullptr);
}

void MPPI::set_elevation(const float* host, MapMeta meta)
{
    const size_t cells = (size_t)meta.nx * meta.ny * 3;
    elev_data_.resize(cells);
    elev_data_.upload_async(host, cells, stream_);
    elev_meta_ = meta;
    has_elev_ = true;
}

MapView MPPI::esdf_view() const
{
    // Dışarıdan verilen cihaz işaretçisi varsa O kullanılır: K9(b)'nin bütün noktası, gridin bu
    // sınıfın kendi tamponuna kopyalanmadan tüketilmesidir.
    if (!has_esdf_) return {nullptr, esdf_meta_};
    return {esdf_external_ != nullptr ? esdf_external_ : esdf_data_.get(), esdf_meta_};
}

MapView MPPI::elev_view() const
{
    return {has_elev_ ? elev_data_.get() : nullptr, elev_meta_};
}

void MPPI::enqueue(float3 x0, const float* ref_host, int N)
{
    submit_(x0, ref_host, N);
}

bool MPPI::poll() const
{
    // SORGU, bekleme değil. cudaErrorNotReady bir hata değil, "henüz değil" cevabıdır.
    const cudaError_t status = cudaEventQuery(done_);
    if (status == cudaErrorNotReady) return false;
    CUDA_CHECK(status);
    return true;
}

void MPPI::wait() const
{
    CUDA_CHECK(cudaEventSynchronize(done_));
}

float2 MPPI::result() const
{
    return {out_pinned_[0], out_pinned_[1]};
}

float2 MPPI::step(float3 x0, const float* ref_host, int N)
{
    // Geriye dönük uyum: submit + bekle + oku. Bekleme artık CİHAZ GENELİ bir bariyer değil, tek bir
    // event üzerinde; yani bu çağrı bloklarken bile diğer stream'ler durmaz.
    submit_(x0, ref_host, N);
    wait();
    const float2 control = result();
    u_prev_ = control;
    return control;
}

void MPPI::submit_(float3 x0, const float* ref_host, int N)
{
    if (N < 2) throw std::invalid_argument("reference window: N >= 2 gerekli");
    const int K = cfg_.K, H = cfg_.H;

    // reference window H2D
    if (N > ref_cap_) {
        ref_.resize((size_t)N * 3);
        ref_cap_ = N;
    }
    ref_.upload_async(ref_host, (size_t)N * 3, stream_);
    const RefWindow ref{ref_.get(), N};

    // 1) SAMPLE: U = U_nom + eps, clamp
    sample_controls(U_nom_.get(), U_.get(), reinterpret_cast<curandState*>(rng_.get()),
                    cfg_.sigma_v, cfg_.sigma_omega,
                    cfg_.robot.v_max, cfg_.robot.omega_max, K, H, stream_);

    // 2) ROLLOUT: S[k] = Σ_t step_cost
    rollout_cost(U_.get(), x0, u_prev_, S_.get(), traj_.get(),
                 cfg_.robot, cfg_.slip, cfg_.weights, cfg_.cost,
                 ref, esdf_view(), elev_view(), cfg_.dt, K, H, stream_);

    // 3) BASELINE: rho = min_k S[k]
    reduce_min(S_.get(), rho_.get(), K, stream_);

    // 4) WEIGHTS: wgt = exp(-(1/λ)(S-ρ)), eta = Σ wgt
    compute_weights(S_.get(), rho_.get(), cfg_.lambda,
                    wgt_.get(), eta_partial_.get(), eta_.get(), K, stream_);

    // 5) UPDATE: U_nom = (1/eta) Σ_k wgt[k]·U[k,·]
    weighted_update(U_.get(), wgt_.get(), eta_.get(), U_nom_.get(), K, H, stream_);
    CUDA_CHECK(cudaGetLastError());

    // 6) OUTPUT: control = U_nom[0] → PINNED host, ASENKRON. Eski hâlde bu satır step()'in %77'siydi
    //    (ölçüldü); artık host burada durmuyor, sıra event'te kuruluyor.
    U_nom_.download_async(out_pinned_, 2, stream_);

    // 7) WARMSTART: U_nom[0..H-2] ← U_nom[1..H-1]; U_nom[H-1] ← U_nom[H-2]
    //
    //    SIRA ÖNEMLİ: warm-start kaydırması, yukarıdaki D2H'den SONRA kuyruğa girer ve aynı
    //    stream'de olduğu için onun arkasında sıralanır. Yani okunan değer kaydırma öncesi
    //    U_nom[0]'dır - eski senkron hâlle birebir aynı değer.
    CUDA_CHECK(cudaMemcpyAsync(shift_.get(), U_nom_.get() + 2,
                               (size_t)(H - 1) * 2 * sizeof(float), cudaMemcpyDeviceToDevice,
                               stream_));
    CUDA_CHECK(cudaMemcpyAsync(U_nom_.get(), shift_.get(),
                               (size_t)(H - 1) * 2 * sizeof(float), cudaMemcpyDeviceToDevice,
                               stream_));
    CUDA_CHECK(cudaMemcpyAsync(U_nom_.get() + (size_t)(H - 1) * 2,
                               U_nom_.get() + (size_t)(H - 2) * 2, 2 * sizeof(float),
                               cudaMemcpyDeviceToDevice, stream_));

    // Tamamlanma işareti. poll() bunu sorgular; step() bunu bekler.
    CUDA_CHECK(cudaEventRecord(done_, stream_));
}

void MPPI::nominal(float* out_h2) const
{
    U_nom_.download(out_h2, (size_t)cfg_.H * 2);
}

void MPPI::last_rollouts(float* out_kh3) const
{
    traj_.download(out_kh3, (size_t)cfg_.K * cfg_.H * 3);
}
