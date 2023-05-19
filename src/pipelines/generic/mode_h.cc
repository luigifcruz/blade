#define BL_LOG_DOMAIN "P::MODE_H"

#include "blade/pipelines/generic/mode_h.hh"

namespace Blade::Pipelines::Generic {

template<typename IT, typename OT>
ModeH<IT, OT>::ModeH(const Config& config)
     : Pipeline(config.accumulateRate, config.detectorIntegrationSize),
       config(config) {
    BL_DEBUG("Initializing Pipeline Mode H.");

    BL_DEBUG("Allocating pipeline buffers.");
    const auto accumulationFactor = ArrayShape({1, 1, config.accumulateRate, 1});
    this->input = ArrayTensor<Device::CUDA, IT>(config.inputShape * accumulationFactor);

    if constexpr (!std::is_same<IT, CF32>::value) {
        BL_DEBUG("Instantiating input cast from {} to CF32.", TypeInfo<IT>::name);
        this->connect(cast, {
            .blockSize = config.castBlockSize,
        }, {
            .buf = this->input,
        });
    }

    BL_DEBUG("Instantiating channelizer with rate {}.", config.inputShape.numberOfTimeSamples() *  
                                                        config.accumulateRate);
    this->connect(channelizer, {
        .rate = config.inputShape.numberOfTimeSamples() * config.accumulateRate,
        .blockSize = config.channelizerBlockSize,
    }, {
        .buf = this->getChannelizerInput(),
    });

    BL_DEBUG("Instatiating polarizer module.")
    this->connect(polarizer, {
        .mode = (config.polarizerConvertToCircular) ? Polarizer::Mode::XY2LR : Polarizer::Mode::BYPASS, 
        .blockSize = config.polarizerBlockSize,
    }, {
        .buf = channelizer->getOutputBuffer(),
    });

    BL_DEBUG("Instantiating detector module.");
    this->connect(detector, {
        .integrationSize = config.detectorIntegrationSize,
        .numberOfOutputPolarizations = config.detectorNumberOfOutputPolarizations,

        .blockSize = config.detectorBlockSize,
    }, {
        .buf = polarizer->getOutputBuffer(),
    });
}

template<typename IT, typename OT>
Result ModeH<IT, OT>::accumulate(const ArrayTensor<Device::CUDA, IT>& data,
                                 const cudaStream_t& stream) {
    const auto& width = (config.inputShape.numberOfTimeSamples() * config.inputShape.numberOfPolarizations()) * sizeof(IT);
    const auto& height = config.inputShape.numberOfAspects() * config.inputShape.numberOfFrequencyChannels();

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
