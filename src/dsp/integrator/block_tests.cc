#include <any>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <blade/config.hh>
#include <blade/integrator/block.hh>
#include <jetstream/domains/core/ones_tensor/block.hh>
#include <jetstream/flowgraph.hh>
#include <jetstream/flowgraph_view.hh>
#include <jetstream/registry.hh>

using namespace Jetstream;

namespace {

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

TEST_CASE("Integrator block declares its module",
          "[blade][integrator][block][registration]") {
    const auto registrations = Registry::ListAvailableBlocks("integrator");
    REQUIRE(registrations.size() == 1);

    const std::vector<Registry::BlockModuleRequirement> expected = {
        {"integrator"},
    };
    REQUIRE(registrations.front().moduleRequirements == expected);
}

TEST_CASE("Integrator preserves invalid block candidates for recovery",
          "[blade][integrator][block][reconfigure]") {
#if !defined(BLADE_DEVICE_CPU_AVAILABLE)
    SKIP("BLADE was not built with CPU support.");
#else
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
#endif
}
