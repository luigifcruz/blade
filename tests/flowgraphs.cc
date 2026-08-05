#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <jetstream/backend/base.hh>
#include <jetstream/domains/core/ones_tensor/block.hh>
#include <jetstream/flowgraph.hh>
#include <jetstream/flowgraph_view.hh>
#include <jetstream/module.hh>
#include <jetstream/registry.hh>

#include <blade/channelizer/block.hh>
#include <blade/config.hh>
#include <blade/integrator/block.hh>
#include <blade/integrator/module.hh>

namespace {

using namespace Jetstream;

struct GraphCase {
    const char* file;
    U64 cycles;
    U64 emissionPeriod;
};

constexpr std::array<GraphCase, 7> kGraphCases = {{
    {"beamformer.yml", 3, 1},
    {"channelizer.yml", 3, 1},
    {"correlator.yml", 6, 2},
    {"detector.yml", 3, 1},
    {"integrator.yml", 6, 3},
    {"polarizer.yml", 3, 1},
    {"stacker.yml", 6, 3},
}};

struct BackendGuard {
    ~BackendGuard() {
        Backend::DestroyAll();
    }
};

struct FlowgraphGuard {
    Flowgraph flowgraph;
    bool created = false;

    ~FlowgraphGuard() {
        destroy();
    }

    Result destroy() {
        if (!created) {
            return Result::SUCCESS;
        }

        Result result = Result::SUCCESS;
        std::vector<std::string> names;
        if (flowgraph.view().keys(names) != Result::SUCCESS) {
            result = Result::ERROR;
        } else {
            for (const auto& name : names) {
                if (flowgraph.blockDestroy(name, false) != Result::SUCCESS) {
                    result = Result::ERROR;
                }
            }
        }
        if (flowgraph.destroy() != Result::SUCCESS) {
            result = Result::ERROR;
        }
        created = false;
        return result;
    }
};

bool HasImplementation(const std::string& type, DeviceType device) {
    const auto implementations = Registry::ListAvailableModules(type);
    return std::any_of(implementations.begin(), implementations.end(), [device](const auto& impl) {
        return impl.device == device &&
               impl.runtime == RuntimeType::NATIVE &&
               impl.provider == "generic";
    });
}

#if defined(BLADE_DEVICE_CPU_AVAILABLE) && defined(BLADE_DEVICE_CUDA_AVAILABLE) && \
    defined(JETSTREAM_BACKEND_CUDA_AVAILABLE)
std::string Metric(const Flowgraph::View::BlockData& block, const std::string& name) {
    const auto metric = std::find_if(block.metrics.begin(), block.metrics.end(),
                                     [&name](const auto& entry) {
                                         return entry.name == name;
                                     });
    REQUIRE(metric != block.metrics.end());
    return std::any_cast<std::string>(metric->value);
}

void MarkComparisonPending(Tensor& error) {
    REQUIRE(error.size() > 0);
    REQUIRE((error.dtype() == DataType::F32 || error.dtype() == DataType::F64));

    if (error.dtype() == DataType::F32) {
        error.data<F32>()[0] = std::numeric_limits<F32>::quiet_NaN();
    } else {
        error.data<F64>()[0] = std::numeric_limits<F64>::quiet_NaN();
    }
}

bool ComparisonCompleted(const Tensor& error) {
    if (error.dtype() == DataType::F32) {
        return std::isfinite(error.data<F32>()[0]);
    }
    if (error.dtype() == DataType::F64) {
        return std::isfinite(error.data<F64>()[0]);
    }
    return false;
}
#endif

}  // namespace

TEST_CASE("BLADE blocks expose only selected device implementations", "[registry]") {
    const auto channelizerRegistrations =
        Registry::ListAvailableBlocks("channelizer");
    REQUIRE(channelizerRegistrations.size() == 1);
    const std::vector<Registry::BlockModuleRequirement> channelizerRequirements = {
        {"invert"},
        {"fft"},
        {"reshape"},
    };
    REQUIRE(channelizerRegistrations.front().moduleRequirements ==
            channelizerRequirements);

    constexpr std::array<const char*, 7> kCpuBlocks = {
        "beamformer",
        "correlator",
        "detector",
        "integrator",
        "phasor",
        "polarizer",
        "stacker",
    };
    constexpr std::array<const char*, 6> kCudaBlocks = {
        "beamformer",
        "correlator",
        "detector",
        "integrator",
        "polarizer",
        "stacker",
    };

    for (const char* type : kCpuBlocks) {
#if defined(BLADE_DEVICE_CPU_AVAILABLE)
        INFO("Missing selected CPU implementation for " << type);
        REQUIRE(HasImplementation(type, DeviceType::CPU));
#else
        INFO("Unexpected unselected CPU implementation for " << type);
        REQUIRE_FALSE(HasImplementation(type, DeviceType::CPU));
#endif
    }

    for (const char* type : kCudaBlocks) {
#if defined(BLADE_DEVICE_CUDA_AVAILABLE)
        INFO("Missing selected CUDA implementation for " << type);
        REQUIRE(HasImplementation(type, DeviceType::CUDA));
#else
        INFO("Unexpected unselected CUDA implementation for " << type);
        REQUIRE_FALSE(HasImplementation(type, DeviceType::CUDA));
#endif
    }
}

TEST_CASE("Channelizer matches the shifted FFT tensor contract",
          "[flowgraph][channelizer][cpu]") {
    BackendGuard backendGuard;
    Backend::Config backendConfig;
    backendConfig.headless = true;
    REQUIRE(Backend::Initialize<DeviceType::CPU>(backendConfig) == Result::SUCCESS);

    FlowgraphGuard graph;
    REQUIRE(graph.flowgraph.create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);
    graph.created = true;

    Blocks::OnesTensor source;
    source.shape = {1, 2, 4, 1};
    source.dataType = "CF32";
    REQUIRE(graph.flowgraph.blockCreate("source", source, {}) == Result::SUCCESS);
    Flowgraph::View::BlockData sourceBlock;
    REQUIRE(graph.flowgraph.view().block("source", sourceBlock) == Result::SUCCESS);
    REQUIRE(sourceBlock.outputs.at("buffer").tensor.setAttribute(
        "sampleRate", F32{1024.0f}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("source", "buffer");

    Blocks::Channelizer channelizer;
    REQUIRE(graph.flowgraph.blockCreate("channelizer", channelizer, inputs) ==
            Result::SUCCESS);
    REQUIRE(graph.flowgraph.compute() == Result::SUCCESS);

    Flowgraph::View::BlockData block;
    REQUIRE(graph.flowgraph.view().block("channelizer", block) == Result::SUCCESS);
    const Tensor output = block.outputs.at("buffer").tensor;
    REQUIRE(output.shape() == Shape{1, 8, 1, 1});
    REQUIRE(output.dtype() == DataType::CF32);
    REQUIRE(std::any_cast<Index>(output.attribute("sampleAxis")) == Index{2});
    REQUIRE(std::any_cast<Index>(output.attribute("channelAxis")) == Index{1});
    REQUIRE(std::any_cast<F32>(output.attribute("sampleRate")) == F32{256.0f});

    constexpr std::array<F32, 8> expected = {0.0f, 0.0f, 4.0f, 0.0f,
                                              0.0f, 0.0f, 4.0f, 0.0f};
    const CF32* outputData = output.data<CF32>();
    REQUIRE(outputData != nullptr);
    for (U64 i = 0; i < expected.size(); ++i) {
        REQUIRE_THAT(outputData[i].real(),
                     Catch::Matchers::WithinAbs(expected[i], 1e-5f));
        REQUIRE_THAT(outputData[i].imag(),
                     Catch::Matchers::WithinAbs(0.0f, 1e-5f));
    }

    REQUIRE(graph.destroy() == Result::SUCCESS);
}

TEST_CASE("Integrator preserves invalid block candidates for recovery",
          "[flowgraph][integrator][reconfigure]") {
    FlowgraphGuard graph;
    REQUIRE(graph.flowgraph.create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);
    graph.created = true;

    Blocks::OnesTensor source;
    source.shape = {1, 1, 4, 1};
    source.dataType = "CF32";
    REQUIRE(graph.flowgraph.blockCreate("source", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("source", "buffer");

    Blocks::Integrator integrator;
    integrator.size = 2;
    integrator.axis = 2;
    REQUIRE(graph.flowgraph.blockCreate("integrator", integrator, inputs) ==
            Result::SUCCESS);

    Parser::Map invalid;
    invalid["size"] = U64{0};
    REQUIRE(graph.flowgraph.blockReconfigure("integrator", invalid) ==
            Result::SUCCESS);

    Flowgraph::View::BlockData block;
    REQUIRE(graph.flowgraph.view().block("integrator", block) == Result::SUCCESS);
    REQUIRE(block.state == Block::State::Errored);
    REQUIRE(block.outputs.empty());

    Parser::Map saved;
    REQUIRE(graph.flowgraph.blockConfig("integrator", saved) == Result::SUCCESS);
    REQUIRE(std::any_cast<U64>(saved.at("size")) == 0);

    Parser::Map recovery;
    recovery["size"] = integrator.size;
    REQUIRE(graph.flowgraph.blockReconfigure("integrator", recovery) ==
            Result::SUCCESS);
    REQUIRE(graph.flowgraph.view().block("integrator", block) == Result::SUCCESS);
    REQUIRE(block.state == Block::State::Created);
    REQUIRE(block.outputs.at("buffer").tensor.shape() == Shape{1, 1, 2, 1});

    REQUIRE(graph.destroy() == Result::SUCCESS);
}

TEST_CASE("Integrator module reconfiguration is transactional",
          "[module][integrator][reconfigure]") {
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
}

TEST_CASE("CPU reference flowgraphs match CUDA", "[flowgraph][cuda][parity]") {
#if !defined(BLADE_DEVICE_CPU_AVAILABLE) || !defined(BLADE_DEVICE_CUDA_AVAILABLE) || \
    !defined(JETSTREAM_BACKEND_CUDA_AVAILABLE)
    SKIP("BLADE was not built with both CPU and CUDA support.");
#else
    BackendGuard backendGuard;
    Backend::Config backendConfig;
    backendConfig.headless = true;
    REQUIRE(Backend::Initialize<DeviceType::CPU>(backendConfig) == Result::SUCCESS);
    REQUIRE(Backend::Initialize<DeviceType::CUDA>(backendConfig) == Result::SUCCESS);

    for (const auto& graphCase : kGraphCases) {
        INFO("Flowgraph: " << graphCase.file);

        FlowgraphGuard graph;
        REQUIRE(graph.flowgraph.create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);
        graph.created = true;
        const std::string path =
            std::string(BLADE_TEST_FLOWGRAPH_DIR) + "/" + graphCase.file;
        REQUIRE(graph.flowgraph.importFromFile(path) == Result::SUCCESS);

        Flowgraph::View::BlockData comparator;
        REQUIRE(graph.flowgraph.view().block("compare", comparator) == Result::SUCCESS);
        Tensor comparisonError = comparator.outputs.at("error").tensor;

        for (U64 cycle = 0; cycle < graphCase.cycles; ++cycle) {
            INFO("Cycle: " << cycle);
            const bool expectsEmission = (cycle + 1) % graphCase.emissionPeriod == 0;
            if (expectsEmission) {
                MarkComparisonPending(comparisonError);
            }

            REQUIRE(graph.flowgraph.compute() == Result::SUCCESS);

            REQUIRE(graph.flowgraph.view().block("compare", comparator) == Result::SUCCESS);
            INFO(comparator.diagnostic);
            REQUIRE(comparator.state == Block::State::Created);
            if (expectsEmission) {
                REQUIRE(ComparisonCompleted(comparisonError));
                REQUIRE(graph.flowgraph.view().metrics("compare", comparator.metrics) ==
                        Result::SUCCESS);
                INFO("Max difference: " << Metric(comparator, "maxDiff"));
                REQUIRE(Metric(comparator, "match") == "PASS");
            }
        }

        REQUIRE(graph.destroy() == Result::SUCCESS);
    }
#endif
}
