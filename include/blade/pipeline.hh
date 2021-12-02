#ifndef BLADE_PIPELINE_HH
#define BLADE_PIPELINE_HH

#include <span>
#include <vector>

#include "blade/common.hh"
#include "blade/logger.hh"
#include "blade/manager.hh"

namespace Blade {

class BLADE_API Pipeline {
 public:
    Pipeline(const bool& async = true, const bool& test = false);
    virtual ~Pipeline();

    Result synchronize();
    bool isSyncronized();

    constexpr Resources getResources() const {
        return resources;
    }

 protected:
    Result setup();
    Result loop();

    virtual constexpr Result setupModules() {
        return Result::SUCCESS;
    }

    virtual constexpr Result setupMemory() {
        return Result::SUCCESS;
    }

    virtual constexpr Result setupTest() {
        return Result::SUCCESS;
    }

    virtual constexpr Result loopPreprocess() {
        return Result::SUCCESS;
    }

    virtual constexpr Result loopUpload() {
        return Result::SUCCESS;
    }

    virtual constexpr Result loopProcess(cudaStream_t& cudaStream) {
        return Result::SUCCESS;
    }

    virtual constexpr Result loopDownload() {
        return Result::SUCCESS;
    }

    virtual constexpr Result loopTest() {
        return Result::SUCCESS;
    }

    template<typename DT, typename ST>
    Result copyBuffer(std::span<DT>& dst, const std::span<ST>& src, CopyKind dir) {
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

    template<typename DT, typename ST>
    Result copyBuffer2D(std::span<DT>& dst, size_t dpitch, const std::span<ST>& src, size_t width, CopyKind dir) {
        if (width > dpitch) {
            BL_FATAL("2D copy 'width' is larger than destination's pitch ({}, {}).",
                    width, dpitch);
            return Result::ASSERTION_ERROR;
        }
        if (dst.size() != dpitch*(src.size_bytes()/width)) {
            BL_FATAL("Destination's size is not exactly covered by {} rows of {} ({} vs {}).",
                    (src.size_bytes()/width), dpitch, dst.size(), dpitch*(src.size_bytes()/width));
            return Result::ASSERTION_ERROR;
        }
        if (width % sizeof(ST) != 0) {
            BL_FATAL("2D copy 'width' is not a multiple of source's element size ({}, {}).",
                    width, sizeof(ST));
            return Result::ASSERTION_ERROR;
        }
        if (src.size() % width != 0) {
            BL_FATAL("2D copy 'width' is not a factor of source's size ({}, {}).",
                    width, src.size());
            return Result::ASSERTION_ERROR;
        }

        BL_CUDA_CHECK(
            cudaMemcpy2DAsync(
                dst.data(),
                dpitch,
                src.data(),
                width,
                width,
                src.size_bytes()/width,
                static_cast<cudaMemcpyKind>(dir), cudaStream),
            [&]{
                BL_FATAL("Can't copy data: {}", err);
            }
        );

        return Result::SUCCESS;
    }

    template<typename T>
    Result allocateBuffer(std::span<T>& dst, std::size_t size, bool managed = false) {
        BL_DEBUG("Allocating device memory.");

        T *ptr;
        std::size_t size_bytes = size * sizeof(ptr[0]);

        if (managed) {
            resources.device += size_bytes;
            resources.host += size_bytes;

            BL_CUDA_CHECK(cudaMallocManaged(&ptr, size_bytes), [&]{
                BL_FATAL("Failed to allocate managed memory: {}", err);
            });
        } else {
            resources.device += size_bytes;

            BL_CUDA_CHECK(cudaMalloc(&ptr, size_bytes), [&]{
                BL_FATAL("Failed to allocate memory: {}", err);
            });
        }

        allocations.push_back(ptr);
        dst = std::span(ptr, size);

        return Result::SUCCESS;
    }

    template<typename T>
    Result pinBuffer(const std::span<T>& mem, RegisterKind kind) {
        BL_DEBUG("Pinning host memory.");

        resources.host += mem.size_bytes();

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

 private:
    bool asyncMode;
    bool testMode;

    cudaGraph_t graph;
    cudaStream_t cudaStream;
    cudaGraphExec_t instance;

    Resources resources;
    std::size_t state{0};
    std::vector<void*> allocations;
};

}  // namespace Blade

#endif
