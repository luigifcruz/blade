#include "blade-cli/telescopes/ata/base.hh"

namespace Blade::CLI::Telescopes::ATA {

Result Setup(const Config& config) {
    switch (config.mode) {
#if defined(BLADE_BUNDLE_ATA_MODE_B)
        case ModeId::MODE_B:
            return ModeB<CI8, CF32>(config);
#endif
#if defined(BLADE_BUNDLE_ATA_MODE_B) && defined(BLADE_BUNDLE_GENERIC_MODE_H)
#endif
        default:
            BL_FATAL("This ATA mode is not implemented yet.");
    }

    return Result::ERROR;
}

}  // namespace Blade::CLI::Telescopes::ATA
