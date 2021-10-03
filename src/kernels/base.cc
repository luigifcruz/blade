#include "blade/kernels/base.hh"

namespace Blade::Kernel {

Manager& Manager::reset() {
    master.memory.host = 0;
    master.memory.device = 0;
    master.transfer.d2h = 0;
    master.transfer.h2d = 0;
    return *this;
}

Manager& Manager::save(const Resources & resources) {
    master.memory.host += resources.memory.host;
    master.memory.device += resources.memory.device;
    master.transfer.d2h += resources.transfer.d2h;
    master.transfer.h2d += resources.transfer.h2d;
    return *this;
}

Manager& Manager::report() {
    BL_INFO("=============================================");
    BL_INFO("Kernel resources manager usage report:")
    BL_INFO("=============================================");
    BL_INFO("Manager configuration:");
    BL_INFO("   PCIe bandwidth:          {} GB/s", toGB(config.pcie_bw));
    BL_INFO("   Device memory bandwidth: {} GB/s", toGB(config.device_bw));
    BL_INFO("   Host memory bandwidth:   {} GB/s", toGB(config.host_bw));
    BL_INFO("Memory usage:");
    BL_INFO("   Host:   {} MB", toMB(master.memory.host));
    BL_INFO("   Device: {} MB", toMB(master.memory.device));
    BL_INFO("Estimated transfers:");
    BL_INFO("   D2H: {} MB @ {} GB/s = {:.1f} ms", toMB(master.transfer.d2h), toGB(config.pcie_bw),
            toMs(master.transfer.d2h, config.pcie_bw));
    BL_INFO("   H2D: {} MB @ {} GB/s = {:.1f} ms", toMB(master.transfer.h2d), toGB(config.pcie_bw),
            toMs(master.transfer.h2d, config.pcie_bw));
    BL_INFO("=============================================");

    return *this;
}

} // namespace Blade::Kernel
