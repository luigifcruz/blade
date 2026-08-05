#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <blade/stacker/block.hh>
#include <jetstream/registry.hh>

using namespace Jetstream;

TEST_CASE("Stacker block declares its module",
          "[blade][stacker][block][registration]") {
    const auto registrations = Registry::ListAvailableBlocks("stacker");
    REQUIRE(registrations.size() == 1);

    const std::vector<Registry::BlockModuleRequirement> expected = {
        {"stacker"},
    };
    REQUIRE(registrations.front().moduleRequirements == expected);
}
