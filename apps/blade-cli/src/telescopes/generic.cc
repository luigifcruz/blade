#include "blade-cli/telescopes/generic/base.hh"

namespace Blade::CLI::Telescopes::Generic {

const Result Setup(const Config& config) {
    switch (config.mode) {
#if defined(BLADE_PIPELINE_ATA_MODE_H)
        case ModeId::MODE_H:
            // return SetupGenericModeH<IT, OT>(config, readerRunner);
#endif
        default:
            BL_FATAL("This generic mode is not implemented yet.");
    }

    return Result::ERROR;
}

const Result CollectUserInput(int argc, char **argv, Config& config) {
    ::CLI::App app("BLADE (Breakthrough Listen Accelerated DSP Engine) - Command Line Tool - Generic");

    return Result::SUCCESS;
}

}  // namespace Blade::CLI::Telescopes::Generic
