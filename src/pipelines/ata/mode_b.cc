#include "blade/pipelines/ata/mode_b.hh"

namespace Blade::Pipelines::ATA {

template<typename OT>
ModeB<OT>::ModeB(const Config& config) : config(config) {
    if ((config.outputMemPad % sizeof(OT)) != 0) {
        BL_FATAL("The outputMemPad must be a multiple of the output type bytes.")
        BL_CHECK_THROW(Result::ASSERTION_ERROR);
    }

    outputMemPitch = config.outputMemPad + config.outputMemWidth;

    BL_DEBUG("Instantiating input cast from I8 to CF32.");
    this->connect(inputCast, {
        .inputSize = config.numberOfBeams *
                     config.numberOfAntennas *
                     config.numberOfFrequencyChannels *
                     config.numberOfTimeSamples *
                     config.numberOfPolarizations,
        .blockSize = config.castBlockSize,
    }, {
        .buf = input,
    });

    BL_DEBUG("Instantiating channelizer with rate {}.", config.channelizerRate);
    this->connect(channelizer, {
        .numberOfBeams = config.numberOfBeams,
        .numberOfAntennas = config.numberOfAntennas,
        .numberOfFrequencyChannels = config.numberOfFrequencyChannels,
        .numberOfTimeSamples = config.numberOfTimeSamples,
        .numberOfPolarizations = config.numberOfPolarizations,
        .rate = config.channelizerRate,
        .blockSize = config.channelizerBlockSize,
    }, {
        .buf = inputCast->getOutput(),
    });

    BL_DEBUG("Instantiating phasor module.");
    this->connect(phasor, {
        .numberOfBeams = config.numberOfBeams * config.beamformerBeams,
        .numberOfAntennas = config.numberOfAntennas,
        .numberOfFrequencyChannels = config.numberOfFrequencyChannels * config.channelizerRate,
        .numberOfPolarizations = config.numberOfPolarizations,

        .rfFrequencyHz = config.rfFrequencyHz,
        .channelBandwidthHz = config.channelBandwidthHz,
        .totalBandwidthHz = config.totalBandwidthHz,
        .frequencyStartIndex = config.frequencyStartIndex,
        .referenceAntennaIndex = config.referenceAntennaIndex,
        .arrayReferencePosition = config.arrayReferencePosition,
        .boresightCoordinate = config.boresightCoordinate,

        .antennaPositions = config.antennaPositions,
        .antennaCalibrations = config.antennaCalibrations,
        .beamCoordinates = config.beamCoordinates,

        .blockSize = config.phasorsBlockSize,
    }, {
        .frameJulianDate = this->frameJulianDate,
        .differenceUniversalTime1 = this->differenceUniversalTime1,
    });

    BL_DEBUG("Instantiating beamformer module.");
    this->connect(beamformer, {
        .numberOfBeams = config.numberOfBeams * config.beamformerBeams,
        .numberOfAntennas = config.numberOfAntennas,
        .numberOfFrequencyChannels = config.numberOfFrequencyChannels * config.channelizerRate,
        .numberOfTimeSamples = config.numberOfTimeSamples / config.channelizerRate,
        .numberOfPolarizations = config.numberOfPolarizations,
        .blockSize = config.beamformerBlockSize,
    }, {
        .buf = channelizer->getOutput(),
        .phasors = phasor->getPhasors(),
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
Result ModeB<OT>::run(const F64& frameJulianDate,
                      const F64& differenceUniversalTime1,
                      const Vector<Device::CPU, CI8>& input,
                            Vector<Device::CPU, OT>& output) {
    this->frameJulianDate = frameJulianDate;
    this->differenceUniversalTime1 = differenceUniversalTime1;

    BL_CHECK(this->copy(inputCast->getInput(), input));
    BL_CHECK(this->compute());
    BL_CHECK(this->copy2D(
        output,
        outputMemPitch,         // dpitch
        this->getOutput(),      // src
        config.outputMemWidth,  // spitch
        config.outputMemWidth,  // width
        (beamformer->getOutputSize()*sizeof(OT))/config.outputMemWidth));

    return Result::SUCCESS;
}

template<typename OT>
Result ModeB<OT>::setPhasors(const Vector<Device::CPU, CF32>& phasors) {
    BL_CHECK(this->copy(beamformer->getPhasors(), phasors));

    return Result::SUCCESS;
}

template class BLADE_API ModeB<CF16>;
template class BLADE_API ModeB<CF32>;

}  // namespace Blade::Pipelines::ATA
