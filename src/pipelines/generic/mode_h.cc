#define BL_LOG_DOMAIN "P::MODE_H"

#include "blade/pipelines/generic/mode_h.hh"

namespace Blade::Pipelines::Generic {

template<typename IT, typename OT>
ModeH<IT, OT>::ModeH(const Config& config)
     : Accumulator(config.accumulateRate),
       config(config) {
    BL_DEBUG("Initializing Pipeline Mode H.");

    if constexpr (!std::is_same<IT, CF32>::value) {
        BL_DEBUG("Instantiating input cast from {} to CF32.", TypeInfo<IT>::name);
        this->connect(cast, {
            .inputSize = this->getInputSize(),
            .blockSize = config.castBlockSize,
        }, {
            .buf = this->input,
        });
    }

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
        .buf = this->getChannelizerInput(),
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

template<typename IT, typename OT>
const Result ModeH<IT, OT>::accumulate(const Vector<Device::CUDA, IT>& data,
                                       const cudaStream_t& stream) {
    // TODO: Check if this copy parameters are correct.
    const auto& width = (data.size() / config.channelizerNumberOfBeams / config.channelizerNumberOfFrequencyChannels) * sizeof(IT);
    const auto& height = config.channelizerNumberOfBeams * config.channelizerNumberOfFrequencyChannels;

    BL_CHECK(
        Memory::Copy2D(
            this->input,
            width * this->getAccumulatorNumberOfSteps(),
            0,
            data,
            width,
            0,
            width,
            height, 
            stream));

    return Result::SUCCESS;
}

template class BLADE_API ModeH<CF16, F32>;
template class BLADE_API ModeH<CF32, F32>;

}  // namespace Blade::Pipelines::Generic
