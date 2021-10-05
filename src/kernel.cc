#include "blade/kernel.hh"

namespace Blade {

Kernel::Kernel() {
/*
    if ((block.x + block.y + block.z) > 1024) {
        BL_FATAL("Block dimension is larger than hardware limit (1024).");
        throw Result::ERROR;
    }

    if (((block.x + block.y + block.z) % 32) != 0) {
        BL_WARN("Best performance is achieved when TBLOCK is a multiple of 32.");
    }
*/
}

} // namespace Blade
