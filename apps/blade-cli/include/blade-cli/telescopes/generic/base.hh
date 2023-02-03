#ifndef BLADE_CLI_TELESCOPES_GENERIC_HH
#define BLADE_CLI_TELESCOPES_GENERIC_HH

#include <chrono>
#include <CLI/CLI.hpp>

#include "blade-cli/types.hh"

#include "blade-cli/telescopes/generic/config.hh"

#if defined(BLADE_PIPELINE_GENERIC_MODE_H)
#endif

namespace Blade::CLI::Telescopes::Generic {

const Result CollectUserInput(int argc, char **argv, Config& config);
const Result Setup(const Config& config);

}  // Blade::CLI::Telecopes::Generic

#endif
