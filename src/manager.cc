#include "blade/manager.hh"

namespace Blade {

Manager& Manager::reset() {
    master.host = 0;
    master.device = 0;
    return *this;
}

Manager& Manager::save(const Resources& resources) {
    master.host += resources.host;
    master.device += resources.device;
    return *this;
}

Manager& Manager::report() {
    BL_INFO("=============================================");
    BL_INFO("Pipeline resources manager usage report:")
    BL_INFO("=============================================");
    BL_INFO("Memory usage:");
    BL_INFO("   Host:   {} MB", toMB(master.host));
    BL_INFO("   Device: {} MB", toMB(master.device));
    BL_INFO("=============================================");

    return *this;
}

}  // namespace Blade
