#ifndef BLADE_MODULE_HH
#define BLADE_MODULE_HH

#include <map>
#include <string>
#include <typeindex>

#include "blade/types.hh"
#include "blade/logger.hh"
#include "blade/macros.hh"

#include "blade/utils/jitify2.hh"
using namespace jitify2::reflection;

namespace Blade {

class Module {
 public:
    explicit Module(const jitify2::PreprocessedProgram& program)
        : cache(100, *program) {};
    virtual ~Module() = default;

    virtual constexpr const Result preprocess(const cudaStream_t& stream, 
                                              const U64& currentComputeCount) {
        return Result::SUCCESS;
    }

    virtual constexpr const Result process(const cudaStream_t& stream) {
        return Result::SUCCESS;
    }

 protected:
    const Result createKernel(const std::string& name,
                              const std::string& key,
                              const dim3& gridSize,
                              const dim3& blockSize, 
                              const auto... templateArguments) {
        if (blockSize.x > 1024) {
            BL_FATAL("The block size ({}, {}, {}) is larger than hardware limit (1024).",
                    blockSize.x, blockSize.y, blockSize.z);
            return Result::ERROR;
        }

        if ((blockSize.x % 32) != 0) {
            BL_WARN("Best performance is achieved when the block size ({}, {}, {}) "
                    "is a multiple of 32.", blockSize.x, blockSize.y, blockSize.z);
        }

        kernels.insert({name, {
            .gridSize = gridSize,
            .blockSize = blockSize,
            .key = Template(key).instantiate(templateArguments...), 
        }});

        return Result::SUCCESS;
    } 

    const Result runKernel(const std::string& name,
                           const cudaStream_t& stream,
                           auto... kernelArguments) {
        const auto& kernel = kernels[name];

        cache
            .get_kernel(kernel.key)
            ->configure(kernel.gridSize, kernel.blockSize, 0, stream)
            ->launch(kernelArguments...);

        BL_CUDA_CHECK_KERNEL([&]{
            BL_FATAL("Module failed to execute: {}", err);
            return Result::CUDA_ERROR;
        });

        return Result::SUCCESS;
    }

    static dim3 PadGridSize(const dim3& gridSize, const dim3& blockSize) {
        return dim3((gridSize.x + (blockSize.x - 1)) / blockSize.x,
                    (gridSize.y + (blockSize.y - 1)) / blockSize.y,
                    (gridSize.z + (blockSize.z - 1)) / blockSize.z);
    }

 private:
    struct Kernel {
        dim3 gridSize;
        dim3 blockSize;
        std::string key;
    };

    jitify2::ProgramCache<> cache;
    std::unordered_map<std::string, Kernel> kernels;
};

}  // namespace Blade

#endif
