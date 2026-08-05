#include <algorithm>
#include <any>
#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <jetstream/backend/base.hh>
#include <jetstream/flowgraph.hh>
#include <jetstream/flowgraph_view.hh>

#include <blade/config.hh>

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
