#ifndef BLADE_PIPELINE_H
#define BLADE_PIPELINE_H

#include "blade/common.hh"
#include "blade/cuda.hh"
#include "blade/logger.hh"
#include "blade/macros.hh"
#include "blade/manager.hh"

namespace Blade {

class BLADE_API Pipeline : public ResourcesPlug {
public:
    virtual ~Pipeline();

    Result commit();
    Result process(bool waitCompletion = false);

    template<typename T>
    Result Register(const std::span<T> & mem, RegisterKind kind) {
        BL_DEBUG("Registering host memory.");

        BL_CUDA_CHECK(cudaHostRegister(mem.data(), mem.size_bytes(),
                    static_cast<unsigned int>(kind)), [&]{
            BL_FATAL("Failed to register host memory: {}", err);
        });

        return Result::SUCCESS;
    }

    template<typename T>
    Result Register(std::vector<T> & mem, RegisterKind kind) {
        return Register(std::span{ mem }, kind);
    }

protected:
    virtual Result underlyingProcess(cudaStream_t & cudaStream) = 0;
    virtual Result underlyingAllocate() = 0;

    template<typename T>
    Result Transfer(const std::span<T> & dst, const std::span<T> & src, CopyKind dir) {
        if (dst.size() != src.size()) {
            BL_FATAL("Size mismatch between source and destination ({}, {}).",
                    src.size(), dst.size());
            return Result::ASSERTION_ERROR;
        }

        BL_CUDA_CHECK(cudaMemcpyAsync(dst.data(), src.data(), src.size_bytes(),
                    static_cast<cudaMemcpyKind>(dir), cudaStream), [&]{
            BL_FATAL("Can't copy data: {}", err);
        });

        return Result::SUCCESS;
    }

    template<typename T>
    Result Allocate(std::size_t size, std::span<T> & dst, bool managed = false) {
        BL_DEBUG("Allocating device memory.");

        T* ptr;
        std::size_t size_bytes = size * sizeof(ptr[0]);

        if (managed) {
            BL_CUDA_CHECK(cudaMallocManaged(&ptr, size_bytes), [&]{
                BL_FATAL("Failed to allocate managed memory: {}", err);
            });
        } else {
            BL_CUDA_CHECK(cudaMalloc(&ptr, size_bytes), [&]{
                BL_FATAL("Failed to allocate memory: {}", err);
            });
        }

        dst = std::span(ptr, size);

        return Result::SUCCESS;
    }

    template<typename T>
    void Free(std::span<T> mem) {
        BL_DEBUG("Freeing device memory.");

        cudaFree(mem.data());
    }

private:
    cudaGraph_t graph;
    cudaStream_t cudaStream;
    cudaGraphExec_t instance;

    bool commited{ false };
};


} // namespace Blade

#endif
