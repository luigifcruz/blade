#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <blade/detector/block.hh>
#include <jetstream/registry.hh>

using namespace Jetstream;

TEST_CASE("Detector block declares its module",
          "[blade][detector][block][registration]") {
    const auto registrations = Registry::ListAvailableBlocks("detector");
    REQUIRE(registrations.size() == 1);

    const std::vector<Registry::BlockModuleRequirement> expected = {
        {"detector"},
    };
    REQUIRE(registrations.front().moduleRequirements == expected);
}
