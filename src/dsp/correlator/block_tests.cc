#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <blade/correlator/block.hh>
#include <jetstream/registry.hh>

using namespace Jetstream;

TEST_CASE("Correlator block declares its module",
          "[blade][correlator][block][registration]") {
    const auto registrations = Registry::ListAvailableBlocks("correlator");
    REQUIRE(registrations.size() == 1);

    const std::vector<Registry::BlockModuleRequirement> expected = {
        {"correlator"},
    };
    REQUIRE(registrations.front().moduleRequirements == expected);
}
