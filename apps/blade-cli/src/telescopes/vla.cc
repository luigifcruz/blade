#include "blade-cli/telescopes/vla/base.hh"

namespace Blade::CLI::Telescopes::VLA {

const Result Setup(const Config& config) {
    switch (config.mode) {
#if defined(BLADE_PIPELINE_VLA_MODE_B)
        case ModeId::MODE_B:
            return ModeB<CI8, CF32>(config);
#endif
        default:
            BL_FATAL("This VLA mode is not implemented yet.");
    }

    return Result::ERROR;
}

}  // namespace Blade::CLI::Telescopes::VLA
