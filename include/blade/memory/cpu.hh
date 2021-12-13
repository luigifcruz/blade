#ifndef BLADE_MEMORY_CPU_HH
#define BLADE_MEMORY_CPU_HH

#include "blade/memory/types.hh"
#include "blade/memory/vector.hh"

namespace Blade {

template<typename T>
class Vector<Device::CPU, T> : public VectorImpl<T> {
 public:
    using VectorImpl<T>::VectorImpl;

    explicit Vector(const std::size_t& size) {
        BL_CHECK_THROW(this->resize(size));
    }

    ~Vector() {
        if (this->container.empty() || !this->managed) {
            return;
        }

        if (cudaFreeHost(this->container.data()) != cudaSuccess) {
            BL_FATAL("Failed to deallocate host memory.");
        }
    }

    // TODO: Implement resize.
    Result resize(const std::size_t& size) override {
        if (!this->container.empty() && !this->managed) {
            return Result::ERROR;
        }

        T* ptr;
        auto size_bytes = size * sizeof(T);

        BL_CUDA_CHECK(cudaMallocHost(&ptr, size_bytes), [&]{
            BL_FATAL("Failed to allocate CPU memory: {}", err);
        });

        this->container = std::span<T>(ptr, size);
        this->managed = true;

        return Result::SUCCESS;
    }

    Result pinMemory(const bool& readOnly = false) {
        cudaPointerAttributes attr;
        BL_CUDA_CHECK(cudaPointerGetAttributes(&attr,
                this->container.data()), [&]{
            BL_FATAL("Failed to get pointer attributes: {}", err);
        });

        if (attr.type != cudaMemoryTypeUnregistered) {
            BL_WARN("Memory already registered.");
            return Result::SUCCESS;
        }

        unsigned int kind = cudaHostRegisterDefault;
        if (readOnly) {
            kind = cudaHostRegisterReadOnly;
        }

        BL_CUDA_CHECK(cudaHostRegister(this->container.data(),
                this->container.size_bytes(), kind), [&]{
            BL_FATAL("Failed to register CPU memory: {}", err);
        });

        return Result::SUCCESS;
    }
};

}  // namespace Blade

#endif
