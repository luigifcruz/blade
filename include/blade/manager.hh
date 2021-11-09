#ifndef BLADE_MANAGER_HH
#define BLADE_MANAGER_HH

#include "blade/common.hh"
#include "blade/logger.hh"

namespace Blade {

struct Resources {
    std::size_t device = 0;
    std::size_t host = 0;
};

class BLADE_API Manager {
 public:
    Manager() {}

    Manager& save(const Resources &resources);
    Manager& reset();
    Manager& report();

    constexpr Resources getResources() const {
        return master;
    }

 protected:
    Resources master;

    std::size_t toMB(const std::size_t& size) {
        return size / 1e6;
    }
};

}  // namespace Blade

#endif
