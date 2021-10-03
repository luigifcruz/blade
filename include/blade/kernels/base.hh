#ifndef BLADE_KERNELS_H
#define BLADE_KERNELS_H

#include "blade/base.hh"

#include "blade/utils/jitify2.hh"
using namespace jitify2::reflection;

namespace Blade::Kernel {

struct Resources {
    struct {
        std::size_t device = 0;
        std::size_t host = 0;
    } memory;

    struct {
        std::size_t d2h = 0;
        std::size_t h2d = 0;
    } transfer;
};

class BLADE_API Generic {
public:
};

class BLADE_API Manager {
public:
    struct Config {
        std::size_t pcie_bw = 22e9;
        std::size_t device_bw = 800e9;
        std::size_t host_bw = 30e9;
    };

    Manager() {};
    Manager(const Config & config) : config(config) {};

    Manager& save(const Resources & resources);
    Manager& reset();
    Manager& report();

protected:
    const Config config;
    Resources master;

    std::size_t toMB(const std::size_t & size) {
        return size / 1e6;
    }

    std::size_t toGB(const std::size_t & size) {
        return size / 1e9;
    }

    float toMs(const std::size_t & part, const std::size_t & whole) {
        return (static_cast<float>(part) / static_cast<float>(whole)) * 1000.0f;
    }
};

} // namespace Blade::Kernel

#endif
