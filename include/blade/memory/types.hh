#ifndef BLADE_MEMORY_TYPES_HH
#define BLADE_MEMORY_TYPES_HH

#include <span>
#include <vector>

#include "blade/logger.hh"
#include "blade/types.hh"

namespace Blade {

enum class Device : uint8_t {
    CPU     = 1 << 0,
    CUDA    = 1 << 1,
    Metal   = 1 << 2,
    Vulkan  = 1 << 3,
};

inline constexpr const Device operator|(Device lhs, Device rhs) {
    return static_cast<Device>(static_cast<uint8_t>(lhs) | static_cast<uint8_t>(rhs));
}

}  // namespace Blade

#endif
