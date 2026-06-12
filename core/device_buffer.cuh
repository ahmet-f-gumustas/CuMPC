#pragma once
#include <cstddef>
#include <utility>

#include "core/cuda_check.cuh"

// RAII device buffer: cudaMalloc/cudaFree sahipliği, kopyalama yok, move OK.
template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t n) { reset(n); }
    ~DeviceBuffer() { free_(); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : ptr_(std::exchange(other.ptr_, nullptr)), n_(std::exchange(other.n_, 0)) {}
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            free_();
            ptr_ = std::exchange(other.ptr_, nullptr);
            n_   = std::exchange(other.n_, 0);
        }
        return *this;
    }

    T* get() const { return ptr_; }
    size_t size() const { return n_; }

    // mevcut içerik korunmaz; n=0 → serbest bırak
    void reset(size_t n) {
        free_();
        if (n > 0) {
            CUDA_CHECK(cudaMalloc(&ptr_, n * sizeof(T)));
            n_ = n;
        }
    }

    void upload(const T* host, size_t n) {
        CUDA_CHECK(cudaMemcpy(ptr_, host, n * sizeof(T), cudaMemcpyHostToDevice));
    }
    void download(T* host, size_t n) const {
        CUDA_CHECK(cudaMemcpy(host, ptr_, n * sizeof(T), cudaMemcpyDeviceToHost));
    }
    void zero() {
        if (ptr_) CUDA_CHECK(cudaMemset(ptr_, 0, n_ * sizeof(T)));
    }

private:
    void free_() {
        if (ptr_) {
            cudaFree(ptr_);
            ptr_ = nullptr;
            n_ = 0;
        }
    }
    T* ptr_ = nullptr;
    size_t n_ = 0;
};
