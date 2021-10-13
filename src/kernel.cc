#include "blade/kernel.hh"

namespace Blade {

Kernel::Kernel(const std::size_t & blockSize) {
    if (blockSize > 1024) {
        BL_FATAL("The block size ({}) is larger than hardware limit (1024).", blockSize);
        throw Result::ERROR;
    }

    if ((blockSize % 32) != 0) {
        BL_WARN("Best performance is achieved when the block size ({}) is a multiple of 32.", blockSize);
    }
}

} // namespace Blade
