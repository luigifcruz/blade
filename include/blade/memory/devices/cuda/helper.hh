#ifndef BLADE_MEMORY_CUDA_HELPER_HH
#define BLADE_MEMORY_CUDA_HELPER_HH

#include "blade/memory/types.hh"
#include "blade/memory/vector.hh"

namespace Blade::Memory {

template<typename Type, typename Dims>
static const Result PageLock(const Vector<Device::CPU, Type, Dims>& vec,
                             const bool& readOnly = false) {
    cudaPointerAttributes attr;
    BL_CUDA_CHECK(cudaPointerGetAttributes(&attr, vec.data()), [&]{
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

    BL_CUDA_CHECK(cudaHostRegister(vec.data(), vec.size_bytes(), kind), [&]{
        BL_FATAL("Failed to register CPU memory: {}", err);
    });

    return Result::SUCCESS;
}

}  // namespace Blade::Memory

#endif
