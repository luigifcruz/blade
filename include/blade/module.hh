#ifndef BLADE_MODULE_HH
#define BLADE_MODULE_HH

#include <string>

#include "blade/common.hh"
#include "blade/logger.hh"
#include "blade/memory.hh"

#include "blade/utils/jitify2.hh"
using namespace jitify2::reflection;

namespace Blade {

class BLADE_API Module {
 public:
    explicit Module(const std::size_t& blockSize,
                    const jitify2::PreprocessedProgram& kernel)
            : cache(100, *kernel) {
        if (blockSize > 1024) {
            BL_FATAL("The block size ({}) is larger than hardware limit (1024).",
                    blockSize);
            throw Result::ERROR;
        }

        if ((blockSize % 32) != 0) {
            BL_WARN("Best performance is achieved when the block size ({}) "
                    "is a multiple of 32.", blockSize);
        }
    }
    virtual ~Module() {}

    virtual constexpr Result preprocess(const cudaStream_t& stream = 0) {
        return Result::SUCCESS;
    }

    virtual constexpr Result process(const cudaStream_t& stream = 0) {
        return Result::SUCCESS;
    }

 protected:
    jitify2::ProgramCache<> cache;
    std::string kernel;
    dim3 grid, block;
};

}  // namespace Blade

#endif
