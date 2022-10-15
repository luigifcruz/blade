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

}  // namespace Blade::CLI::Telescopes::Generic
