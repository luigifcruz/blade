#ifndef BL_HELPERS_H
#define BL_HELPERS_H

#include "bl-beamformer/type.hh"
#include "bl-beamformer/logger.hh"
#include "bl-beamformer/tools/magic_enum.hh"
#include "bl-beamformer/tools/jitify2.hh"
using namespace jitify2::reflection;

#ifndef BL_CUDA_CHECK_KERNEL
#define BL_CUDA_CHECK_KERNEL() { \
    cudaError_t err; \
    if ((err = cudaPeekAtLastError()) != cudaSuccess) { \
        BL_FATAL("Kernel failed to execute: {}", cudaGetErrorString(err)); \
        return Result::CUDA_ERROR; \
    } \
}
#endif

#ifndef BL_CUDA_CHECK
#define BL_CUDA_CHECK(x, callback) { \
    cudaError_t val = (x); \
    if (val != cudaSuccess) { \
        callback(); \
        return Result::CUDA_ERROR; \
    } \
}
#endif

#ifndef BL_CHECK
#define BL_CHECK(x) { \
    Result val = (x); \
    if (val != Result::SUCCESS) { \
        return val; \
    } \
}
#endif

#ifndef BL_ASSERT
#define BL_ASSERT(x) { \
    bool val = (x); \
    if (val != true) { \
        return Result::ASSERTION_ERROR; \
    } \
}
#endif

namespace BL {

class BL_API Helpers {
public:
    static Result LoadFromFile(const char* filename, void* cudaMemory, std::size_t size, std::size_t len);
    static Result PrintState();
};

} // namespace BL::Helpers

#endif
