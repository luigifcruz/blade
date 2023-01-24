#include "blade-cli/telescopes/ata/base.hh"

namespace Blade::CLI::Telescopes::ATA {

const Result Setup(const Config& config) {
    switch (config.mode) {
#if defined(BLADE_PIPELINE_ATA_MODE_B)
        case ModeId::MODE_B:
            switch (config.outputType) {
                case TypeId::CF16:
                    return ModeB<CI8, CF16>(config);
                case TypeId::CF32:
                    return ModeB<CI8, CF32>(config);
                case TypeId::F16:
                    return ModeB<CI8, F16>(config);
                case TypeId::F32:
                    return ModeB<CI8, F32>(config);
                default:
                    BL_FATAL("This ATA output is not implemented yet.");    
            }
#endif
#if defined(BLADE_PIPELINE_GENERIC_MODE_H) && defined(BLADE_PIPELINE_ATA_MODE_B) && defined(BLADE_PIPELINE_GENERIC_MODE_S)
        case ModeId::MODE_BS:
            return ModeBS<CI8>(config);
#endif
#if defined(BLADE_PIPELINE_ATA_MODE_B) && defined(BLADE_PIPELINE_GENERIC_MODE_H)
#endif
        default:
            BL_FATAL("This ATA mode is not implemented yet.");
    }

    return Result::ERROR;
}

}  // namespace Blade::CLI::Telescopes::ATA
