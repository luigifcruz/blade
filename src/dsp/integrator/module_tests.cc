#include <algorithm>
#include <any>
#include <cstddef>
#include <memory>
#include <string>

#include <catch2/catch_test_macros.hpp>

#include <blade/config.hh>
#include <blade/integrator/module.hh>
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

TEST_CASE("Integrator module exposes selected implementations",
          "[blade][integrator][module][registration]") {
    const auto implementations = Registry::ListAvailableModules("integrator");
    std::size_t expected = 0;

#if defined(BLADE_DEVICE_CPU_AVAILABLE)
    REQUIRE(HasImplementation("integrator", DeviceType::CPU));
    ++expected;
#else
    REQUIRE_FALSE(HasImplementation("integrator", DeviceType::CPU));
#endif

#if defined(BLADE_DEVICE_CUDA_AVAILABLE)
    REQUIRE(HasImplementation("integrator", DeviceType::CUDA));
    ++expected;
#else
    REQUIRE_FALSE(HasImplementation("integrator", DeviceType::CUDA));
#endif

    REQUIRE(implementations.size() == expected);
    for (const auto& implementation : implementations) {
        DYNAMIC_SECTION("Device: " << implementation.device
                                    << " Runtime: " << implementation.runtime) {
            std::shared_ptr<Module> module;
            REQUIRE(Registry::BuildModule("integrator",
                                          implementation.device,
                                          implementation.runtime,
                                          implementation.provider,
                                          module) == Result::SUCCESS);
            REQUIRE(module != nullptr);
        }
    }
}

TEST_CASE("Integrator module reconfiguration is transactional",
          "[blade][integrator][module][reconfigure]") {
#if !defined(BLADE_DEVICE_CPU_AVAILABLE)
    SKIP("BLADE was not built with CPU support.");
#else
    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("integrator",
                                  DeviceType::CPU,
                                  RuntimeType::NATIVE,
                                  "generic",
                                  module) == Result::SUCCESS);

    Tensor input;
    REQUIRE(input.create(DeviceType::CPU, DataType::CF32, {1, 1, 4, 1}) ==
            Result::SUCCESS);
    TensorMap inputs;
    inputs["buffer"].requested("source", "buffer");
    inputs["buffer"].tensor = input;

    Modules::Integrator config;
    REQUIRE(module->create("integrator", config, inputs) == Result::SUCCESS);
    REQUIRE(module->state() == Module::State::CREATED);

    Parser::Map invalid;
    invalid["size"] = U64{0};
    REQUIRE(module->reconfigure(invalid) == Result::ERROR);
    REQUIRE(module->state() == Module::State::CREATED);

    Parser::Map saved;
    REQUIRE(module->config(saved) == Result::SUCCESS);
    REQUIRE(std::any_cast<U64>(saved.at("size")) == config.size);
    REQUIRE(module->destroy() == Result::SUCCESS);
#endif
}
