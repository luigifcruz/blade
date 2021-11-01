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
    explicit Kernel(const std::size_t& blockSize);
};

template<typename T>
inline std::unique_ptr<T> Factory(const typename T::Config& config) {
    return std::make_unique<T>(config);
}

}  // namespace Blade

#endif  // BLADE_INCLUDE_BLADE_KERNEL_HH_
