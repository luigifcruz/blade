#ifndef BLADE_MODULE_H
#define BLADE_MODULE_H

#include "blade/common.hh"
#include "blade/logger.hh"

#include "blade/utils/jitify2.hh"
using namespace jitify2::reflection;

namespace Blade {

class module {
 public:
    explicit module(const std::size_t& blockSize);
};

template<typename T>
inline std::unique_ptr<T> Factory(const typename T::Config& config) {
    return std::make_unique<T>(config);
}

}  // namespace Blade

#endif  // BLADE_INCLUDE_BLADE_MODULE_HH_
