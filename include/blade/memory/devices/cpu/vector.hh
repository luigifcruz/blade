#ifndef BLADE_MEMORY_CPU_VECTOR_HH
#define BLADE_MEMORY_CPU_VECTOR_HH

#include "blade/memory/types.hh"
#include "blade/memory/vector.hh"

namespace Blade {

template<typename T, typename Dims>
class BLADE_API Vector<Device::CPU, T, Dims> : public VectorImpl<T, Dims> {
 public:
    using VectorImpl<T, Dims>::VectorImpl;

    explicit Vector(const Dims& dims) {
        BL_CHECK_THROW(this->resize(dims));
    }

    ~Vector() {
        if (this->container.empty() || !this->managed) {
            return;
        }

        if (cudaFreeHost(this->container.data()) != cudaSuccess) {
            BL_FATAL("Failed to deallocate host memory.");
        }
    }

    const Result resize(const Dims& dims) final {
        // TODO: Implement resize.
        if (!this->container.empty()) { 
            return Result::ERROR;
        }

        if (!this->managed) {
            return Result::ERROR;
        }

        // Calculate byte size.
        auto size_bytes = dims.size() * sizeof(T);

        // Allocate memory with CUDA.
        T* ptr;
        BL_CUDA_CHECK(cudaMallocHost(&ptr, size_bytes), [&]{
            BL_FATAL("Failed to allocate CPU memory: {}", err);
        });

        // Register metadata.
        this->container = std::span<T>(ptr, dims.size());
        static_cast<Dims&>(*this) = dims;
        this->managed = true;

        return Result::SUCCESS;
    }
};

}  // namespace Blade

#endif
