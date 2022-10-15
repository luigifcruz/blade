#ifndef BLADE_CLI_BASE_HH
#define BLADE_CLI_BASE_HH

#include <chrono>
#include <CLI/CLI.hpp>

#include "blade-cli/types.hh"
#include "blade-cli/telescopes/ata/base.hh"
#include "blade-cli/telescopes/generic/base.hh"

namespace Blade::CLI {

const Result Start(int argc, char **argv);
const Result CollectUserInput(int argc, char **argv, Config& config);
const Result SetupProcessingPipeline(const Config& config);
const Result SetupTelescope(const Config& config);

}  // namespace Blade::CLI

#endif
