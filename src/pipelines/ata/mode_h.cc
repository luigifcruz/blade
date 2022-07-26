#include "blade/pipelines/ata/mode_h.hh"

namespace Blade::Pipelines::ATA {

template<typename OT>
ModeH<OT>::ModeH(const Config& config) : config(config) {
    BL_DEBUG("Initializing ATA Pipeline Mode H.");

    BL_DEBUG("Instantiating channelizer with rate {}.", config.channelizerNumberOfTimeSamples *  
                                                        config.accumulateRate);
    this->connect(channelizer, {
        .numberOfBeams = config.channelizerNumberOfBeams,
        .numberOfAntennas = 1,
        .numberOfFrequencyChannels = config.channelizerNumberOfFrequencyChannels,
        .numberOfTimeSamples = config.channelizerNumberOfTimeSamples * config.accumulateRate,
        .numberOfPolarizations = config.channelizerNumberOfPolarizations,
        .rate = config.channelizerNumberOfTimeSamples * config.accumulateRate,
        .blockSize = config.channelizerBlockSize,
    }, {
        .buf = input,
    });

    BL_DEBUG("Instantiating detector module.");
    this->connect(detector, {
        .numberOfBeams = config.channelizerNumberOfBeams, 
        .numberOfFrequencyChannels = config.channelizerNumberOfFrequencyChannels * 
                                     config.channelizerNumberOfTimeSamples * 
                                     config.accumulateRate,
        .numberOfTimeSamples = 1,
        .numberOfPolarizations = config.channelizerNumberOfPolarizations,

        .integrationSize = 1,
        .numberOfOutputPolarizations = config.detectorNumberOfOutputPolarizations,

        .blockSize = config.detectorBlockSize,
    }, {
        .buf = channelizer->getOutput(),
    });
}

template<typename OT>
Result ModeH<OT>::run(Vector<Device::CPU, OT>& output) {
    BL_CHECK(this->compute());
    BL_CHECK(this->copy(output, this->getOutput()));

    return Result::SUCCESS;
}

template class BLADE_API ModeH<F32>;

}  // namespace Blade::Pipelines::ATA
