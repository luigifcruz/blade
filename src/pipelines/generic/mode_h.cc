#define BL_LOG_DOMAIN "P::MODE_H"

#include "blade/pipelines/generic/mode_h.hh"

namespace Blade::Pipelines::Generic {

template<typename IT, typename OT>
ModeH<IT, OT>::ModeH(const Config& config)
     : Pipeline(config.accumulateRate, config.detectorIntegrationSize),
       config(config) {
    BL_DEBUG("Initializing Pipeline Mode H.");

    BL_DEBUG("Allocating pipeline buffers.");
    const auto accumulationFactor = ArrayDimensions{1, 1, config.accumulateRate, 1};
    BL_CHECK_THROW(this->input.resize(config.inputDimensions * accumulationFactor));

    if constexpr (!std::is_same<IT, CF32>::value) {
        BL_DEBUG("Instantiating input cast from {} to CF32.", TypeInfo<IT>::name);
        this->connect(cast, {
            .blockSize = config.castBlockSize,
        }, {
            .buf = this->input,
        });
    }

    BL_DEBUG("Instantiating channelizer with rate {}.", config.inputDimensions.numberOfTimeSamples() *  
                                                        config.accumulateRate);
    this->connect(channelizer, {
        .rate = config.inputDimensions.numberOfTimeSamples() * config.accumulateRate,
        .blockSize = config.channelizerBlockSize,
    }, {
        .buf = this->getChannelizerInput(),
    });

    BL_DEBUG("Instantiating detector module.");
    this->connect(detector, {
        .integrationSize = config.detectorIntegrationSize,
        .numberOfOutputPolarizations = config.detectorNumberOfOutputPolarizations,

        .blockSize = config.detectorBlockSize,
    }, {
        .buf = channelizer->getOutputBuffer(),
    });
}

template<typename IT, typename OT>
const Result ModeH<IT, OT>::accumulate(const ArrayTensor<Device::CUDA, IT>& data,
                                       const cudaStream_t& stream) {
    const auto& width = (config.inputDimensions.numberOfTimeSamples() * config.inputDimensions.numberOfPolarizations()) * sizeof(IT);
    const auto& height = config.inputDimensions.numberOfAspects() * config.inputDimensions.numberOfFrequencyChannels();

    BL_CHECK(
        Memory::Copy2D(
            this->input,
            width * this->getAccumulatorNumberOfSteps(),
            width * this->getCurrentAccumulatorStep(),
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
