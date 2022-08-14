#include "blade/pipelines/ata/mode_a.hh"

namespace Blade::Pipelines::ATA {

template<typename OT>
ModeA<OT>::ModeA(const Config& config) : config(config), blockJulianDate(1), blockDut1(1) {
    BL_DEBUG("Initializing ATA Pipeline Mode A.");

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
        .blockJulianDate = this->blockJulianDate,
        .blockDut1 = this->blockDut1,
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
        .enableIncoherentBeamSqrt = true,
        .blockSize = config.beamformerBlockSize,
    }, {
        .buf = channelizer->getOutput(),
        .phasors = phasor->getPhasors(),
    });

    BL_DEBUG("Instantiating detector module.");
    this->connect(detector, {
        .numberOfBeams = config.beamformerNumberOfBeams + 
                         (config.beamformerIncoherentBeam ? 1 : 0), 
        .numberOfFrequencyChannels = config.beamformerNumberOfFrequencyChannels * 
                                     config.preBeamformerChannelizerRate,
        .numberOfTimeSamples = config.beamformerNumberOfTimeSamples / 
                               config.preBeamformerChannelizerRate,
        .numberOfPolarizations = config.beamformerNumberOfPolarizations,

        .integrationSize = config.detectorIntegrationSize,
        .numberOfOutputPolarizations = config.detectorNumberOfOutputPolarizations,

        .blockSize = config.detectorBlockSize,
    }, {
        .buf = beamformer->getOutput(),
    });
}

template<typename OT>
const Result ModeA<OT>::run(const Vector<Device::CPU, F64>& blockJulianDate,
                            const Vector<Device::CPU, F64>& blockDut1,
                            const Vector<Device::CPU, CI8>& input,
                                    Vector<Device::CPU, OT>& output) {
    if (this->getCurrentComputeStep() == 0) {
        BL_DEBUG("Block Julian Date: {}", blockJulianDate[0]);
        BL_DEBUG("Block DUT1: {}", blockDut1[0]);
    }

    BL_CHECK(this->copy(this->blockJulianDate, blockJulianDate));
    BL_CHECK(this->copy(this->blockDut1, blockDut1));
    BL_CHECK(this->copy(inputCast->getInput(), input));
    BL_CHECK(this->compute());
    BL_CHECK(this->copy(output, this->getOutput()));

    return Result::SUCCESS;
}

template class BLADE_API ModeA<F32>;

}  // namespace Blade::Pipelines::ATA
