#ifndef BLADE_MEMORY_TYPES_HH
#define BLADE_MEMORY_TYPES_HH

#include <span>
#include <vector>

#include "blade/logger.hh"
#include "blade/types.hh"

namespace Blade {

struct Device {
    class CPU;
    class CUDA;
    class Metal;
    class Vulkan;
};

struct Unified {
    class CPU;
    class CUDA;
    class Metal;
    class Vulkan;
};

}  // namespace Blade

#endif
