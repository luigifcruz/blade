#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <blade/polarizer/block.hh>
#include <jetstream/registry.hh>

using namespace Jetstream;

TEST_CASE("Polarizer block declares its module",
          "[blade][polarizer][block][registration]") {
    const auto registrations = Registry::ListAvailableBlocks("polarizer");
    REQUIRE(registrations.size() == 1);

    const std::vector<Registry::BlockModuleRequirement> expected = {
        {"polarizer"},
    };
    REQUIRE(registrations.front().moduleRequirements == expected);
}
