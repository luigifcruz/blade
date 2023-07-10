#ifndef BLADE_MEMORY_HELPER_HH
#define BLADE_MEMORY_HELPER_HH

#include <cuda_runtime.h>

#include "blade/memory/types.hh"
#include "blade/memory/vector.hh"

namespace Blade::Memory {

template<typename Type, typename Dims>
inline Result PageLock(const Vector<Device::CPU, Type, Dims>& vec,
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

template<Device DeviceId, typename Type, typename Shape>
inline Result Link(Vector<DeviceId, Type, Shape>& dst,
                          const Vector<DeviceId, Type, Shape>& src) {
    dst = src;
    return Result::SUCCESS;
}

template<Device DeviceId, typename Type, typename Shape>
inline Result Link(Vector<DeviceId, Type, Shape>& dst,
                          const Vector<DeviceId, Type, Shape>& src,
                          const Shape dstShape) {
    dst = src;
    return dst.reshape(dstShape);
    return Result::SUCCESS;
}

inline std::string ReadableBytes(uint64_t bytes) {
    const double GB = 1e9;
    const double MB = 1e6;
    const double KB = 1e3;

    char buffer[50];

    if (bytes >= GB) {
        sprintf(buffer, "%.2f GB", bytes / GB);
    } else if (bytes >= MB) {
        sprintf(buffer, "%.2f MB", bytes / MB);
    } else if (bytes >= KB) {
        sprintf(buffer, "%.2f KB", bytes / KB);
    } else {
        sprintf(buffer, "%ld bytes", bytes);
    }

    return std::string(buffer);
}

}  // namespace Blade::Memory

#endif
