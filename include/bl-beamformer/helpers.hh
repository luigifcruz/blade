#ifndef BL_HELPERS_H
#define BL_HELPERS_H

#include "bl-beamformer/type.hh"
#include "bl-beamformer/logger.hh"
#include "bl-beamformer/tools/jitify2.hh"
using namespace jitify2::reflection;

namespace BL::Helpers {

#ifndef CUDA_CHECK_KERNEL
#define CUDA_CHECK_KERNEL() { \
    cudaError_t err; \
    if ((err = cudaPeekAtLastError()) != cudaSuccess) { \
        BL_FATAL("Kernel failed to execute: {}", cudaGetErrorString(err)); \
        return Result::ERROR; \
    } \
}
#endif

#ifndef CUDA_CHECK
#define CUDA_CHECK(x, callback) { \
    cudaError_t val = (x); \
    if (val != cudaSuccess) { \
        callback(); \
        return Result::ERROR; \
    } \
}
#endif

#ifndef CHECK
#define CHECK(x) { \
    Result val = (x); \
    if (val != Result::SUCCESS) { \
        return val; \
    } \
}
#endif

Result LoadFromFile(const char* filename, void* cudaMemory, size_t size, size_t len);
Result PrintState();

} // namespace BL::Helpers

#endif
