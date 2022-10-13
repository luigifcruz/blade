#ifndef BLADE_MEMORY_CUDA_VECTOR_HH
#define BLADE_MEMORY_CUDA_VECTOR_HH

#include <memory>

#include "blade/memory/types.hh"
#include "blade/memory/vector.hh"
#include "blade/memory/devices/cuda/copy.hh"

namespace Blade {

template<typename Type, typename Dims>
class BLADE_API Vector<Device::CUDA, Type, Dims>
     : public VectorImpl<Type, Dims> {
 public:
    using VectorImpl<Type, Dims>::VectorImpl;

    explicit Vector(const Dims& dims) : VectorImpl<Type, Dims>(dims) {
        BL_CHECK_THROW(this->resize(dims));
    }

    Vector(const Vector& other) : VectorImpl<Type, Dims>(other.dims()) {
        BL_DEBUG("Vector copy performed ({} bytes) on CUDA.",
                 other.dims().size() * sizeof(Type));
        BL_CHECK_THROW(this->resize(other.dims()));
        BL_CHECK_THROW(Memory::Copy(*this, other));
    }

    ~Vector() {
        if (this->container.empty() || !this->managed) {
            return;
        }

        if (cudaFree(this->container.data()) != cudaSuccess) {
            BL_FATAL("Failed to deallocate CUDA memory.");
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
        BL_CUDA_CHECK(cudaMalloc(&ptr, size_bytes), [&]{
            BL_FATAL("Failed to allocate CUDA memory: {}", err);
        });

        // Register metadata.
        this->container = std::span<Type>(ptr, dims.size());
        this->dimensions = dims;
        this->managed = true;

        return Result::SUCCESS;
    }
};

template<typename Type, typename Dims>
class BLADE_API Vector<Device::CUDA | Device::CPU, Type, Dims>
     : public VectorImpl<Type, Dims> {
 public:
    using VectorImpl<Type, Dims>::VectorImpl;

    explicit Vector(const Dims& dims) : VectorImpl<Type, Dims>(dims) {
        BL_CHECK_THROW(this->resize(dims));
    }

    Vector(const Vector& other) : VectorImpl<Type, Dims>(other.dims()) {
        BL_DEBUG("Vector copy performed ({} bytes) on CUDA.",
                 other.dims().size() * sizeof(Type));
        BL_CHECK_THROW(this->resize(other.dims()));
        BL_CHECK_THROW(Memory::Copy(*this, other));
    }

    ~Vector() {
        if (this->container.empty() || !this->managed) {
            return;
        }

        if (cudaFree(this->container.data()) != cudaSuccess) {
            BL_FATAL("Failed to deallocate CUDA memory.");
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
        BL_CUDA_CHECK(cudaMallocManaged(&ptr, size_bytes), [&]{
            BL_FATAL("Failed to allocate CUDA memory: {}", err);
        });

        // Register metadata.
        this->container = std::span<Type>(ptr, dims.size());
        this->dimensions = dims;
        this->managed = true;

        // Delete unused vector.
        this->cpuVector.release();
        this->cudaVector.release();

        // Recreate CPU binding.
        if (!this->cpuVector) {
            this->cpuVector = std::make_unique
                    <Vector<Device::CPU, Type, Dims>>(this->container, this->dimensions);
        }

        // Recreate CUDA binding.
        if (!this->cudaVector) {
            this->cudaVector = std::make_unique
                    <Vector<Device::CUDA, Type, Dims>>(this->container, this->dimensions);
        }

        return Result::SUCCESS;
    }

    operator Vector<Device::CPU, Type, Dims>&() const {
        return *this->cpuVector;
    }

    operator Vector<Device::CUDA, Type, Dims>&() const {
        return *this->cudaVector;
    }

 protected:
    std::unique_ptr<Vector<Device::CPU, Type, Dims>> cpuVector;
    std::unique_ptr<Vector<Device::CUDA, Type, Dims>> cudaVector;
};

}  // namespace Blade

#endif
