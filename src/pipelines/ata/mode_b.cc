#include "blade/pipelines/ata/mode_b.hh"

namespace Blade::Pipelines::ATA {

template<typename OT>
ModeB<OT>::ModeB(const Config& config) : config(config), frameJulianDate(1), frameDut1(1) {
    BL_DEBUG("Initializing ATA Pipeline Mode B.");

    if ((config.outputMemPad % sizeof(OT)) != 0) {
        BL_FATAL("The outputMemPad must be a multiple of the output type bytes.")
        BL_CHECK_THROW(Result::ASSERTION_ERROR);
    }

    outputMemPitch = config.outputMemPad + config.outputMemWidth;

    BL_DEBUG("Instantiating input cast from I8 to CF32.");
    this->connect(inputCast, {
        .inputSize = config.beamformerNumberOfAntennas *
                     config.beamformerNumberOfFrequencyChannels *
                     config.beamformerNumberOfTimeSamples *
                     config.beamformerNumberOfPolarizations,
        .blockSize = config.castBlockSize,
    }, {
        .buf = input,
    });

    BL_DEBUG("Instantiating pre-beamformer channelizer with rate {}.",
            config.preBeamformerChannelizerRate);
    this->connect(channelizer, {
        .numberOfBeams = 1,
        .numberOfAntennas = config.beamformerNumberOfAntennas,
        .numberOfFrequencyChannels = config.beamformerNumberOfFrequencyChannels,
        .numberOfTimeSamples = config.beamformerNumberOfTimeSamples,
        .numberOfPolarizations = config.beamformerNumberOfPolarizations,
        .rate = config.preBeamformerChannelizerRate,
        .blockSize = config.channelizerBlockSize,
    }, {
        .buf = inputCast->getOutput(),
    });

    BL_DEBUG("Instantiating phasor module.");
    this->connect(phasor, {
        .numberOfBeams = config.beamformerNumberOfBeams,
        .numberOfAntennas = config.beamformerNumberOfAntennas,
        .numberOfFrequencyChannels = config.beamformerNumberOfFrequencyChannels * 
                                     config.preBeamformerChannelizerRate,
        .numberOfPolarizations = config.beamformerNumberOfPolarizations,

        .observationFrequencyHz = config.phasorObservationFrequencyHz,
        .channelBandwidthHz = config.phasorChannelBandwidthHz,
        .totalBandwidthHz = config.phasorTotalBandwidthHz,
        .frequencyStartIndex = config.phasorFrequencyStartIndex,
        .referenceAntennaIndex = config.phasorReferenceAntennaIndex,
        .arrayReferencePosition = config.phasorArrayReferencePosition,
        .boresightCoordinate = config.phasorBoresightCoordinate,

        .antennaPositions = config.phasorAntennaPositions,
        .antennaCalibrations = config.phasorAntennaCalibrations,
        .beamCoordinates = config.phasorBeamCoordinates,

        .blockSize = config.phasorBlockSize,
    }, {
        .frameJulianDate = this->frameJulianDate,
        .frameDut1 = this->frameDut1,
    });

    BL_DEBUG("Instantiating beamformer module.");
    this->connect(beamformer, {
        .numberOfBeams = config.beamformerNumberOfBeams, 
        .numberOfAntennas = config.beamformerNumberOfAntennas,
        .numberOfFrequencyChannels = config.beamformerNumberOfFrequencyChannels * 
                                     config.preBeamformerChannelizerRate,
        .numberOfTimeSamples = config.beamformerNumberOfTimeSamples / 
                               config.preBeamformerChannelizerRate,
        .numberOfPolarizations = config.beamformerNumberOfPolarizations,
        .enableIncoherentBeam = config.beamformerIncoherentBeam,
        .enableIncoherentBeamSqrt = false,
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
                      const F64& frameDut1,
                      const Vector<Device::CPU, CI8>& input,
                            Vector<Device::CPU, OT>& output) {
    this->frameJulianDate[0] = frameJulianDate;
    this->frameDut1[0] = frameDut1;

    if (this->getStepCount() == 0) {
        BL_DEBUG("Frame Julian Date: {}", frameJulianDate);
        BL_DEBUG("Frame DUT1: {}", frameDut1);
    }

    BL_CHECK(this->copy(inputCast->getInput(), input));
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

template<typename OT>
Result ModeB<OT>::run(const F64& frameJulianDate,
                      const F64& frameDut1,
                      const Vector<Device::CPU, CI8>& input, 
                      const U64& outputBlockIndex,
                      const U64& outputNumberOfBlocks,
                            Vector<Device::CUDA, OT>& output) {
    this->frameJulianDate[0] = frameJulianDate;
    this->frameDut1[0] = frameDut1;

    if (this->getStepCount() == 0) {
        BL_DEBUG("Frame Julian Date: {}", frameJulianDate);
        BL_DEBUG("Frame DUT1: {}", frameDut1);
    }

    BL_CHECK(this->copy(inputCast->getInput(), input));
    BL_CHECK(this->compute());

    const auto& width = (beamformer->getOutputSize() / config.beamformerNumberOfBeams / 
            (config.beamformerNumberOfFrequencyChannels * config.preBeamformerChannelizerRate)) * sizeof(OT);
    const auto& height = config.beamformerNumberOfBeams * 
            (config.beamformerNumberOfFrequencyChannels * config.preBeamformerChannelizerRate);

    BL_CHECK(this->copy2D(
        output,
        width * outputNumberOfBlocks,
        0,
        this->getOutput(),
        width,
        0,
        width,
        height
    )); 

    return Result::SUCCESS;
}

template class BLADE_API ModeB<CF16>;
template class BLADE_API ModeB<CF32>;

}  // namespace Blade::Pipelines::ATA
