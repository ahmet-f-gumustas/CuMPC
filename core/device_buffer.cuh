#pragma once
#include <cstddef>
#include <utility>

#include "core/cuda_check.cuh"

// RAII device buffer: cudaMalloc/cudaFree sahipliği, kopyalama yok, move OK.
//
// ---------------------------------------------------------------------------------
// KAPASİTE-MAKSİMUM ÖMÜR MODELİ (BudgetRT runtime sözleşmesi)
// ---------------------------------------------------------------------------------
//
// Önceki hâl `reset(n)` çağrısında ÖNCE serbest bırakıp SONRA yeniden ayırıyordu. Bu iki şeyi aynı
// anda kırıyor:
//
//   1. TICK BAŞINA ALLOCATION. Ölçüldü (BudgetRT `tests/perf/stage_d_task5_planner_costs/`):
//      cost map her tick beslendiğinde tick başına 1 `cudaMalloc` + 1 `cudaFree` (2 MiB) ve bu
//      ikisi tek başına ~103 µs. Warm-up sonrası kritik yolda allocation, çağıran runtime'ın
//      INV-05'i ile doğrudan çelişir.
//   2. ASENKRON KOPYAYLA USE-AFTER-FREE. Kopyalar stream'e taşındığı anda, `free_()` hâlâ kuyrukta
//      bekleyen bir kopyanın altından belleği çekebilir. Bu yüzden stream parametrelendirmesi ile
//      ömür modeli AYNI değişikliktir; biri diğeri olmadan yapılamaz.
//
// Yeni model: `reserve()` bir kez (ve yalnız büyürken) ayırır, `resize()` yalnız AKTİF uzunluğu
// değiştirir. Sıcak yolda hiçbir çağrı serbest bırakmaz. `reset()` geriye dönük uyum için durur ama
// artık küçültmez — eski çağıranların davranışı değişmez, maliyeti kaybolur.
template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t n) { reset(n); }
    ~DeviceBuffer() { free_(); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : ptr_(std::exchange(other.ptr_, nullptr)),
          n_(std::exchange(other.n_, 0)),
          cap_(std::exchange(other.cap_, 0)) {}
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            free_();
            ptr_ = std::exchange(other.ptr_, nullptr);
            n_   = std::exchange(other.n_, 0);
            cap_ = std::exchange(other.cap_, 0);
        }
        return *this;
    }

    T* get() const { return ptr_; }
    size_t size() const { return n_; }
    size_t capacity() const { return cap_; }

    // Kapasiteyi en az `n` yapar. Küçültmez ve mevcut içeriği korumayı VAAT ETMEZ.
    void reserve(size_t n) {
        if (n <= cap_) return;
        free_();
        CUDA_CHECK(cudaMalloc(&ptr_, n * sizeof(T)));
        cap_ = n;
    }

    // Yalnız aktif uzunluğu değiştirir; kapasiteyi aşarsa büyütür. Sıcak yolda kapasite zaten
    // yeterliyse HİÇBİR CUDA çağrısı yapmaz — ölçülen ~103 µs/tick tam olarak burada kayboluyor.
    void resize(size_t n) {
        reserve(n);
        n_ = n;
    }

    // mevcut içerik korunmaz; n=0 → aktif uzunluk sıfırlanır (BELLEK SERBEST BIRAKILMAZ).
    void reset(size_t n) { resize(n); }

    // Belleği gerçekten geri veren tek yol. Sıcak yolda ÇAĞRILMAZ: adı da bunu söylüyor.
    void release() { free_(); }

    void upload(const T* host, size_t n) {
        CUDA_CHECK(cudaMemcpy(ptr_, host, n * sizeof(T), cudaMemcpyHostToDevice));
    }
    void download(T* host, size_t n) const {
        CUDA_CHECK(cudaMemcpy(host, ptr_, n * sizeof(T), cudaMemcpyDeviceToHost));
    }
    void zero() {
        if (ptr_) CUDA_CHECK(cudaMemset(ptr_, 0, n_ * sizeof(T)));
    }

    // Stream'li karşılıkları. `upload_async` PAGEABLE host belleğiyle çağrıldığında CUDA'nın kendisi
    // senkron davranır; bu bir kusur değil, host tamponunun sahibinin pinned vermesi gerektiğini
    // söyleyen bir sözleşmedir ve çağıran taraf bunu bilerek seçer.
    void upload_async(const T* host, size_t n, cudaStream_t stream) {
        CUDA_CHECK(cudaMemcpyAsync(ptr_, host, n * sizeof(T), cudaMemcpyHostToDevice, stream));
    }
    void download_async(T* host, size_t n, cudaStream_t stream) const {
        CUDA_CHECK(cudaMemcpyAsync(host, ptr_, n * sizeof(T), cudaMemcpyDeviceToHost, stream));
    }
    void zero_async(cudaStream_t stream) {
        if (ptr_) CUDA_CHECK(cudaMemsetAsync(ptr_, 0, n_ * sizeof(T), stream));
    }

private:
    void free_() {
        if (ptr_) {
            cudaFree(ptr_);
            ptr_ = nullptr;
            n_ = 0;
            cap_ = 0;
        }
    }
    T* ptr_ = nullptr;
    size_t n_ = 0;
    size_t cap_ = 0;
};
