#include "blade/pipelines/ata/mode_a.hh"

namespace Blade::Pipelines::ATA {

template<typename OT>
ModeA<OT>::ModeA(const Config& config) : config(config), frameJulianDate(1), frameDut1(1) {
    BL_DEBUG("Initializing ATA Pipeline Mode A.");

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

    BL_DEBUG("Instantiating phasor module.");
    this->connect(phasor, {
        .numberOfBeams = config.beamformerBeams,
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
        .frameDut1 = this->frameDut1,
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
        .phasors = phasor->getPhasors(),
    });

    BL_DEBUG("Instantiating detector module.");
    this->connect(detector, {
        .numberOfBeams = config.beamformerBeams, 
        .numberOfFrequencyChannels = config.numberOfFrequencyChannels * config.channelizerRate,
        .numberOfTimeSamples = config.numberOfTimeSamples / config.channelizerRate,
        .numberOfPolarizations = config.numberOfPolarizations,

        .integrationSize = config.integrationSize,
        .numberOfOutputPolarizations = config.numberOfOutputPolarizations,

        .blockSize = config.detectorBlockSize,
    }, {
        .buf = beamformer->getOutput(),
    });
}

template<typename OT>
Result ModeA<OT>::run(const F64& frameJulianDate,
                      const F64& frameDut1,
                      const Vector<Device::CPU, CI8>& input,
                            Vector<Device::CPU, OT>& output) {
    this->frameJulianDate[0] = frameJulianDate;
    this->frameDut1[0] = frameDut1;

    BL_CHECK(this->copy(inputCast->getInput(), input));
    BL_CHECK(this->compute());
    BL_CHECK(this->copy(output, this->getOutput()));

    return Result::SUCCESS;
}

template class BLADE_API ModeA<F32>;

}  // namespace Blade::Pipelines::ATA
