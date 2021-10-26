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

    constexpr Resources getResources() const {
        return resources;
    }

    template<typename T>
    Result pinBuffer(const std::span<T>& mem, RegisterKind kind) {
        BL_DEBUG("Pinning host memory.");

        resources.memory.host += mem.size_bytes();

        BL_CUDA_CHECK(cudaHostRegister(mem.data(), mem.size_bytes(),
                    static_cast<unsigned int>(kind)), [&]{
            BL_FATAL("Failed to register host memory: {}", err);
        });

        return Result::SUCCESS;
    }

    template<typename T>
    Result pinBuffer(std::vector<T>& mem, RegisterKind kind) {
        return pinBuffer(std::span{ mem }, kind);
    }

protected:
    virtual constexpr Result underlyingInit() {
        return Result::SUCCESS;
    }

    virtual constexpr Result underlyingReport(Resources& res) {
        return Result::SUCCESS;
    }

    virtual constexpr Result underlyingAllocate() {
        return Result::SUCCESS;
    }

    virtual constexpr Result underlyingPreprocess() {
        return Result::SUCCESS;
    }

    virtual constexpr Result underlyingProcess(cudaStream_t& cudaStream) {
        return Result::SUCCESS;
    }

    virtual constexpr Result underlyingPostprocess() {
        return Result::SUCCESS;
    }

    template<typename T>
    Result copyBuffer(const std::span<T>& dst, const std::span<T>& src, CopyKind dir) {
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
    Result allocateBuffer(std::span<T>& dst, std::size_t size, bool managed = false) {
        BL_DEBUG("Allocating device memory.");

        T *ptr;
        std::size_t size_bytes = size * sizeof(ptr[0]);

        if (managed) {
            resources.memory.device += size_bytes;
            resources.memory.host += size_bytes;

            BL_CUDA_CHECK(cudaMallocManaged(&ptr, size_bytes), [&]{
                BL_FATAL("Failed to allocate managed memory: {}", err);
            });
        } else {
            resources.memory.device += size_bytes;

            BL_CUDA_CHECK(cudaMalloc(&ptr, size_bytes), [&]{
                BL_FATAL("Failed to allocate memory: {}", err);
            });
        }

        allocations.push_back(ptr);
        dst = std::span(ptr, size);

        return Result::SUCCESS;
    }

private:
    cudaGraph_t graph;
    cudaStream_t cudaStream;
    cudaGraphExec_t instance;

    Resources resources;
    std::vector<void*> allocations;
};

} // namespace Blade

#endif
