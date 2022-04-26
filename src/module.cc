#include "blade/module.hh"

namespace Blade {

Module::Module(const U64& blockSize,
               const jitify2::PreprocessedProgram& kernel)
        : cache(100, *kernel),
          block(blockSize) {
    if (blockSize > 1024) {
        BL_FATAL("The block size ({}) is larger than hardware limit (1024).",
                blockSize);
        BL_CHECK_THROW(Result::ERROR);
    }

    if ((blockSize % 32) != 0) {
        BL_WARN("Best performance is achieved when the block size ({}) "
                "is a multiple of 32.", blockSize);
    }

    this->block = dim3(blockSize);
}

}  // namespace Blade
