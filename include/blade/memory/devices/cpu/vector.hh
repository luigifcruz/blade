#ifndef BLADE_MEMORY_CPU_VECTOR_HH
#define BLADE_MEMORY_CPU_VECTOR_HH

#include "blade/memory/types.hh"
#include "blade/memory/vector.hh"
#include "blade/memory/devices/cpu/copy.hh"

namespace Blade {

template<typename Type, typename Dims>
class BLADE_API Vector<Device::CPU, Type, Dims>
     : public VectorImpl<Type, Dims> {
 public:
    using VectorImpl<Type, Dims>::VectorImpl;

    explicit Vector(const Dims& dims) : VectorImpl<Type, Dims>(dims) {
        BL_CHECK_THROW(this->resize(dims));
    }

    Vector(const Vector& other) : VectorImpl<Type, Dims>(other.dims()) {
        BL_DEBUG("Vector copy performed ({} bytes) on CPU.",
                 other.dims().size() * sizeof(Type));
        BL_CHECK_THROW(this->resize(other.dims()));
        BL_CHECK_THROW(Memory::Copy(*this, other));
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
        if (!this->container.empty()) {
            BL_FATAL("Can't resize initialized vector.");
            return Result::ERROR;
        }

        if (!this->managed) {
            BL_FATAL("Can't resize non-managed vector.");
            return Result::ERROR;
        }

        // Calculate byte size.
        auto size_bytes = dims.size() * sizeof(Type);

        // Allocate memory with CUDA.
        Type* ptr;
        BL_CUDA_CHECK(cudaMallocHost(&ptr, size_bytes), [&]{
            BL_FATAL("Failed to allocate CPU memory: {}", err);
        });

        // Register metadata.
        this->container = std::span<Type>(ptr, dims.size());
        this->dimensions = dims;
        this->managed = true;

        return Result::SUCCESS;
    }
};

}  // namespace Blade

#endif
