#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <blade/phasor/block.hh>
#include <jetstream/registry.hh>

using namespace Jetstream;

TEST_CASE("Phasor block declares its module",
          "[blade][phasor][block][registration]") {
    const auto registrations = Registry::ListAvailableBlocks("phasor");
    REQUIRE(registrations.size() == 1);

    const std::vector<Registry::BlockModuleRequirement> expected = {
        {"phasor"},
    };
    REQUIRE(registrations.front().moduleRequirements == expected);
}
