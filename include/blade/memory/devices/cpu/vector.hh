#ifndef BLADE_MEMORY_CPU_VECTOR_HH
#define BLADE_MEMORY_CPU_VECTOR_HH

#include "blade/memory/types.hh"
#include "blade/memory/vector.hh"

namespace Blade {

template<typename T>
class BLADE_API Vector<Device::CPU, T> : public VectorImpl<T> {
 public:
    using VectorImpl<T>::VectorImpl;

    explicit Vector(const U64& size) {
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
    Result resize(const U64& size) override {
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
};

}  // namespace Blade

#endif
