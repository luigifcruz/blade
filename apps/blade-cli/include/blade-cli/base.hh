#ifndef BLADE_CLI_BASE_HH
#define BLADE_CLI_BASE_HH

#include <chrono>
#include <CLI/CLI.hpp>

#include "blade-cli/types.hh"
#include "blade-cli/telescopes/ata/base.hh"
#include "blade-cli/telescopes/generic/base.hh"

namespace Blade::CLI {

Result Start(int argc, char **argv);
Result CollectUserInput(int argc, char **argv, Config& config);
Result SetupProcessingPipeline(const Config& config);
Result SetupTelescope(const Config& config);

}  // namespace Blade::CLI

#endif
