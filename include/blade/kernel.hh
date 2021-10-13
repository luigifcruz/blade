#ifndef BLADE_KERNEL_H
#define BLADE_KERNEL_H

#include "blade/common.hh"
#include "blade/types.hh"
#include "blade/logger.hh"

#include "blade/utils/jitify2.hh"
using namespace jitify2::reflection;

namespace Blade {

class BLADE_API Kernel {
public:
    Kernel(const std::size_t & blockSize);
};

} // namespace Blade

#endif
