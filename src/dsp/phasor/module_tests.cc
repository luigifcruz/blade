#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>

#include <catch2/catch_test_macros.hpp>

#include <blade/config.hh>
#include <blade/phasor/module.hh>
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

TEST_CASE("Phasor module exposes its selected CPU implementation",
          "[blade][phasor][module][registration]") {
    const auto implementations = Registry::ListAvailableModules("phasor");
    std::size_t expected = 0;

#if defined(BLADE_DEVICE_CPU_AVAILABLE)
    REQUIRE(HasImplementation("phasor", DeviceType::CPU));
    ++expected;
#else
    REQUIRE_FALSE(HasImplementation("phasor", DeviceType::CPU));
#endif

    REQUIRE_FALSE(HasImplementation("phasor", DeviceType::CUDA));
    REQUIRE(implementations.size() == expected);

    for (const auto& implementation : implementations) {
        DYNAMIC_SECTION("Device: " << implementation.device
                                    << " Runtime: " << implementation.runtime) {
            std::shared_ptr<Module> module;
            REQUIRE(Registry::BuildModule("phasor",
                                          implementation.device,
                                          implementation.runtime,
                                          implementation.provider,
                                          module) == Result::SUCCESS);
            REQUIRE(module != nullptr);
        }
    }
}
