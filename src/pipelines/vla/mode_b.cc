#include "blade/pipelines/vla/mode_b.hh"

namespace Blade::Pipelines::VLA {

template<typename OT>
ModeB<OT>::ModeB(const Config& config) : config(config) {
    BL_DEBUG("Initializing VLA Pipeline Mode B.");

    if ((config.outputMemPad % sizeof(OT)) != 0) {
        BL_FATAL("The outputMemPad must be a multiple of the output type bytes.")
        BL_CHECK_THROW(Result::ASSERTION_ERROR);
    }

    outputMemPitch = config.outputMemPad + config.outputMemWidth;

    BL_DEBUG("Instantiating input cast from I8 to CF32.");
    this->connect(inputCast, {
        .inputSize = config.numberOfAntennas *
                     config.numberOfFrequencyChannels *
                     config.numberOfTimeSamples *
                     config.numberOfPolarizations,
        .blockSize = config.castBlockSize,
    }, {
        .buf = input,
    });

    BL_DEBUG("Instantiating channelizer with rate {}.", config.channelizerRate);
    this->connect(channelizer, {
        .numberOfBeams = 1,
        .numberOfAntennas = config.numberOfAntennas,
        .numberOfFrequencyChannels = config.numberOfFrequencyChannels,
        .numberOfTimeSamples = config.numberOfTimeSamples,
        .numberOfPolarizations = config.numberOfPolarizations,
        .rate = config.channelizerRate,
        .blockSize = config.channelizerBlockSize,
    }, {
        .buf = inputCast->getOutput(),
    });

    BL_DEBUG("Instantiating beamformer module.");
    this->connect(beamformer, {
        .numberOfBeams = config.beamformerBeams,
        .numberOfAntennas = config.numberOfAntennas,
        .numberOfFrequencyChannels = config.numberOfFrequencyChannels * config.channelizerRate,
        .numberOfTimeSamples = config.numberOfTimeSamples / config.channelizerRate,
        .numberOfPolarizations = config.numberOfPolarizations,
        .blockSize = config.beamformerBlockSize,
    }, {
        .buf = channelizer->getOutput(),
        .phasors = phasors,
    });

    if constexpr (!std::is_same<OT, CF32>::value) {
        BL_DEBUG("Instantiating output cast from CF32 to {}.", typeid(OT).name());
        this->connect(outputCast, {
            .inputSize = beamformer->getOutputSize(),
            .blockSize = config.castBlockSize,
        }, {
            .buf = beamformer->getOutput(),
        });
    }
}

template<typename OT>
Result ModeB<OT>::run(const Vector<Device::CPU, CI8>& input,
                      const Vector<Device::CPU, CF32>& phasors,
                            Vector<Device::CPU, OT>& output) {
    BL_CHECK(this->copy(inputCast->getInput(), input));
    BL_CHECK(this->copy(this->phasors, phasors));
    BL_CHECK(this->compute());
    BL_CHECK(this->copy2D(
        output,
        outputMemPitch,         // dpitch
        0,
        this->getOutput(),      // src
        config.outputMemWidth,  // spitch
        0,
        config.outputMemWidth,  // width
        (beamformer->getOutputSize() * sizeof(OT)) / config.outputMemWidth
    ));

    return Result::SUCCESS;
}

template class BLADE_API ModeB<CF16>;
template class BLADE_API ModeB<CF32>;

}  // namespace Blade::Pipelines::VLA
