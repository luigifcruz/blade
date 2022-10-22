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

    template<typename T>
    static const std::string CudaType() {
        static std::unordered_map<std::type_index, std::string> type_map = {
            {typeid(CF16),  "__half"},
            {typeid(CF32),  "float"},
            {typeid(CI8),   "char"},
            {typeid(CI16),  "short"},
            {typeid(CI32),  "long"},
            {typeid(F16),   "__half"},
            {typeid(F32),   "float"},
            {typeid(I8),    "char"},
            {typeid(I16),   "short"},
            {typeid(I32),   "long"},
        };

        auto& type = typeid(T);
        if (!type_map.contains(type)) {
            BL_FATAL("Type is not supported by CudaType.");
            BL_CHECK_THROW(Result::ERROR);
        }
        return type_map[type];
    }

    template<typename T>
    static const std::size_t CudaTypeSize() {
        static std::unordered_map<std::type_index, std::size_t> size_map = {
            {typeid(CF16),  2},
            {typeid(CF32),  2},
            {typeid(CI8),   2},
            {typeid(CI16),  2},
            {typeid(CI32),  2},
            {typeid(F16),   1},
            {typeid(F32),   1},
            {typeid(I8),    1},
            {typeid(I16),   1},
            {typeid(I32),   1},
        };

        auto& type = typeid(T);
        if (!size_map.contains(type)) {
            BL_FATAL("Type is not supported by CudaTypeSize.");
            BL_CHECK_THROW(Result::ERROR);
        }
        return size_map[type];
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
