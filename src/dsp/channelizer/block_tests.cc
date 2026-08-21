#include <any>
#include <array>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <blade/channelizer/block.hh>
#include <jetstream/backend/base.hh>
#include <jetstream/domains/core/ones_tensor/block.hh>
#include <jetstream/flowgraph.hh>
#include <jetstream/flowgraph_view.hh>
#include <jetstream/registry.hh>

using namespace Jetstream;

namespace {

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

}  // namespace

TEST_CASE("Channelizer block declares its composed modules",
          "[blade][channelizer][block][registration]") {
    const auto registrations = Registry::ListAvailableBlocks("channelizer");
    REQUIRE(registrations.size() == 1);

    const std::vector<Registry::BlockModuleRequirement> expected = {
        {"invert"},
        {"fft"},
        {"reshape"},
    };
    REQUIRE(registrations.front().moduleRequirements == expected);
}

TEST_CASE("Channelizer matches the shifted FFT tensor contract",
          "[blade][channelizer][block][cpu]") {
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

TEST_CASE("Channelizer reports its accepted sample-count constraint",
          "[blade][channelizer][block][validation][diagnostic]") {
    BackendGuard backendGuard;
    Backend::Config backendConfig;
    backendConfig.headless = true;
    REQUIRE(Backend::Initialize<DeviceType::CPU>(backendConfig) == Result::SUCCESS);

    FlowgraphGuard graph;
    REQUIRE(graph.flowgraph.create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);
    graph.created = true;

    Blocks::OnesTensor source;
    source.shape = {1, 2, 3, 1};
    source.dataType = "CF32";
    REQUIRE(graph.flowgraph.blockCreate("source", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("source", "buffer");

    Blocks::Channelizer channelizer;
    REQUIRE(graph.flowgraph.blockCreate("channelizer", channelizer, inputs) ==
            Result::SUCCESS);

    Flowgraph::View::BlockData block;
    REQUIRE(graph.flowgraph.view().block("channelizer", block) == Result::SUCCESS);
    REQUIRE(block.state == Block::State::Errored);
    REQUIRE(block.diagnostic ==
            "[BLOCK_CHANNELIZER] Sample count 3 must be one or even.");

    REQUIRE(graph.destroy() == Result::SUCCESS);
}
