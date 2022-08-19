#include "blade/pipelines/ata/mode_b.hh"

namespace Blade::Pipelines::ATA {

template<typename OT>
ModeB<OT>::ModeB(const Config& config) : config(config), blockJulianDate(1), blockDut1(1) {
    BL_DEBUG("Initializing ATA Pipeline Mode B.");

    if ((config.outputMemPad % sizeof(OT)) != 0) {
        BL_FATAL("The outputMemPad must be a multiple of the output type bytes.")
        BL_CHECK_THROW(Result::ASSERTION_ERROR);
    }

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
        .enableIncoherentBeamSqrt = (config.detectorEnable) ? true : false,
        .blockSize = config.beamformerBlockSize,
    }, {
        .buf = channelizer->getOutput(),
        .phasors = phasor->getPhasors(),
    });

    if (config.detectorEnable) {
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

        if constexpr (!std::is_same<OT, F32>::value) {
            BL_DEBUG("Instantiating output cast from F32 to {}.", TypeID<OT>());
            this->connect(outputCast, {
                .inputSize = detector->getOutputSize(),
                .blockSize = config.castBlockSize,
            }, {
                .buf = detector->getOutput(),
            });
        }
    } else {
        if constexpr (!std::is_same<OT, CF32>::value) {
            BL_DEBUG("Instantiating output cast from CF32 to {}.", TypeID<OT>());
            this->connect(complexOutputCast, {
                .inputSize = beamformer->getOutputSize(),
                .blockSize = config.castBlockSize,
            }, {
                .buf = beamformer->getOutput(),
            });
        }
    }
}

template<typename OT>
const Result ModeB<OT>::underlyingRun(const Vector<Device::CPU, F64>& blockJulianDate,
                                      const Vector<Device::CPU, F64>& blockDut1,
                                      const Vector<Device::CPU, CI8>& input) { 
    // Print dynamic arguments on first run.
    if (this->getCurrentComputeStep() == 0) {
        BL_DEBUG("Block Julian Date: {}", blockJulianDate[0]);
        BL_DEBUG("Block DUT1: {}", blockDut1[0]);
    }

    // Copy input to static buffers.
    BL_CHECK(this->copy(this->blockJulianDate, blockJulianDate));
    BL_CHECK(this->copy(this->blockDut1, blockDut1));
    BL_CHECK(this->copy(inputCast->getInput(), input));

    // Trigger pipeline modules compute.
    BL_CHECK(this->compute());

    return Result::SUCCESS;
}

template<typename OT>
const Result ModeB<OT>::run(const Vector<Device::CPU, F64>& blockJulianDate,
                            const Vector<Device::CPU, F64>& blockDut1,
                            const Vector<Device::CPU, CI8>& input,
                                    Vector<Device::CPU, OT>& output) {
    // Print debug messages, copy input variables, and compute.
    BL_CHECK(this->underlyingRun(blockJulianDate, blockDut1, input));

    // Handle the output accordingly.
    if (config.detectorEnable) {
        BL_CHECK(this->copy(output, this->getOutput()));
    } else {
        BL_CHECK(
            this->copy2D(
                output,
                config.outputMemPad + config.outputMemWidth,
                0,
                this->getOutput(),
                config.outputMemWidth,
                0,
                config.outputMemWidth,
                (beamformer->getOutputSize() * sizeof(OT)) / config.outputMemWidth));
    }

    return Result::SUCCESS;
}

template class BLADE_API ModeB<CF32>;
template class BLADE_API ModeB<CF16>;
template class BLADE_API ModeB<F32>;
template class BLADE_API ModeB<F16>;

}  // namespace Blade::Pipelines::ATA
