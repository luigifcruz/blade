#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>

#include <catch2/catch_test_macros.hpp>

#include <blade/beamformer/module.hh>
#include <blade/config.hh>
#include <jetstream/module.hh>
#include <jetstream/registry.hh>

using namespace Jetstream;

namespace {

bool HasImplementation(const std::string& type, const DeviceType device) {
    const auto implementations = Registry::ListAvailableModules(type);
    return std::any_of(implementations.begin(), implementations.end(),
                       [device](const auto& implementation) {
                           return implementation.device == device &&
                                  implementation.runtime == RuntimeType::NATIVE &&
                                  implementation.provider == "generic";
                       });
}

}  // namespace

TEST_CASE("Beamformer module exposes selected implementations",
          "[blade][beamformer][module][registration]") {
    const auto implementations = Registry::ListAvailableModules("beamformer");
    std::size_t expected = 0;

#if defined(BLADE_DEVICE_CPU_AVAILABLE)
    REQUIRE(HasImplementation("beamformer", DeviceType::CPU));
    ++expected;
#else
    REQUIRE_FALSE(HasImplementation("beamformer", DeviceType::CPU));
#endif

#if defined(BLADE_DEVICE_CUDA_AVAILABLE)
    REQUIRE(HasImplementation("beamformer", DeviceType::CUDA));
    ++expected;
#else
    REQUIRE_FALSE(HasImplementation("beamformer", DeviceType::CUDA));
#endif

    REQUIRE(implementations.size() == expected);
    for (const auto& implementation : implementations) {
        DYNAMIC_SECTION("Device: " << implementation.device
                                    << " Runtime: " << implementation.runtime) {
            std::shared_ptr<Module> module;
            REQUIRE(Registry::BuildModule("beamformer",
                                          implementation.device,
                                          implementation.runtime,
                                          implementation.provider,
                                          module) == Result::SUCCESS);
            REQUIRE(module != nullptr);
        }
    }
}
