#ifndef BLADE_MANAGER_H
#define BLADE_MANAGER_H

#include "blade/common.hh"
#include "blade/types.hh"
#include "blade/logger.hh"

namespace Blade {

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

class BLADE_API ResourcesPlug {
public:
    virtual ~ResourcesPlug() {};
    virtual Resources getResources() const = 0;
};

class BLADE_API Manager {
public:
    struct Config {
        std::size_t pcie_bw = 22e9;
    };

    explicit Manager() {};
    Manager(const Config & config) : config(config) {};

    Manager& save(const Resources &resources);
    Manager& save(ResourcesPlug &plug);
    Manager& reset();
    Manager& report();

    constexpr Resources getResources() const {
        return master;
    }

protected:
    const Config config;
    Resources master;

    std::size_t toMB(const std::size_t& size) {
        return size / 1e6;
    }

    std::size_t toGB(const std::size_t& size) {
        return size / 1e9;
    }

    float toMs(const std::size_t& part, const std::size_t& whole) {
        return (static_cast<float>(part) / static_cast<float>(whole)) * 1000.0f;
    }
};

} // namespace Blade

#endif
