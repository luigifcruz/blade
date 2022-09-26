#ifndef BLADE_CLI_TELESCOPES_VLA_HH
#define BLADE_CLI_TELESCOPES_VLA_HH

#include "blade-cli/types.hh"

#if defined(BLADE_PIPELINE_VLA_MODE_B)
#include "blade-cli/telescopes/vla/mode_b.hh"
#endif

namespace Blade::CLI::Telescopes::VLA {

const Result Setup(const Config& config);

}  // namespace Blade::CLI::Telecopes::VLA

#endif
