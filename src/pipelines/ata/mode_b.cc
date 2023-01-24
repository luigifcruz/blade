#define BL_LOG_DOMAIN "P::ATA::MODE_B"

#include "blade/pipelines/ata/mode_b.hh"

namespace Blade::Pipelines::ATA {

template<typename IT, typename OT>
ModeB<IT, OT>::ModeB(const Config& config)
    : Pipeline(config.accumulateRate, 1),
      config(config),
      blockJulianDate({1}),
      blockDut1({1}),
      blockFrequencyChannelOffset({1}) {
    BL_DEBUG("Initializing ATA Pipeline Mode B.");

    BL_DEBUG("Allocating pipeline buffers.");
    // accumulation in time
    const auto accumulationFactor = ArrayDimensions{.A=1, .F=1, .T=config.accumulateRate, .P=1};
    BL_CHECK_THROW(this->input.resize(config.inputDimensions * accumulationFactor));
    if (config.accumulateRate > 1) {
        BL_DEBUG("Input Dimensions: {}", config.inputDimensions);
        BL_DEBUG("Accumulated Input Dimensions: {}", this->input.dims());
        BL_DEBUG("Accumulated Input Byte Size: {}", this->input.size_bytes());
    }

    if constexpr (!std::is_same<IT, CF32>::value) {
        BL_DEBUG("Instantiating input cast from {} to CF32.", TypeInfo<IT>::name);
        this->connect(inputCast, {
            .blockSize = config.castBlockSize,
        }, {
            .buf = this->input,
        });
    }

    BL_DEBUG("Instantiating pre-beamformer channelizer with rate {}.",
            config.preBeamformerChannelizerRate);
    this->connect(channelizer, {
        .rate = config.preBeamformerChannelizerRate,

        .blockSize = config.channelizerBlockSize,
    }, {
        .buf = this->getChannelizerInput(),
    });

    BL_DEBUG("Instatiating polarizer module.")
    this->connect(polarizer, {
        .mode = (config.preBeamformerPolarizerConvertToCircular) 
                    ? Polarizer::Mode::XY2LR : Polarizer::Mode::BYPASS, 
        .blockSize = config.polarizerBlockSize,
    }, {
        .buf = channelizer->getOutputBuffer(),
    });

    BL_DEBUG("Instantiating phasor module.");
    this->connect(phasor, {
        .numberOfAntennas = channelizer->getOutputBuffer().dims().numberOfAspects(),
        .numberOfFrequencyChannels = channelizer->getOutputBuffer().dims().numberOfFrequencyChannels(),
        .numberOfPolarizations = channelizer->getOutputBuffer().dims().numberOfPolarizations(),

        .observationFrequencyHz = config.phasorObservationFrequencyHz,
        .channelBandwidthHz = config.phasorChannelBandwidthHz,
        .totalBandwidthHz = config.phasorTotalBandwidthHz,
        .frequencyStartIndex = config.phasorFrequencyStartIndex,

        .referenceAntennaIndex = config.phasorReferenceAntennaIndex,
        .arrayReferencePosition = config.phasorArrayReferencePosition,
        .boresightCoordinate = config.phasorBoresightCoordinate,
        .antennaPositions = config.phasorAntennaPositions,
        .antennaCoefficients = config.phasorAntennaCoefficients,
        .beamCoordinates = config.phasorBeamCoordinates,

        .antennaCoefficientChannelRate = this->config.phasorAntennaCoefficientChannelRate,

        .blockSize = config.phasorBlockSize,
    }, {
        .blockJulianDate = this->blockJulianDate,
        .blockDut1 = this->blockDut1,
        .blockFrequencyChannelOffset = this->blockFrequencyChannelOffset,
    });

    BL_DEBUG("Instantiating beamformer module.");
    this->connect(beamformer, {
        .enableIncoherentBeam = config.beamformerIncoherentBeam,
        .enableIncoherentBeamSqrt = (config.detectorEnable) ? true : false,

        .blockSize = config.beamformerBlockSize,
    }, {
        .buf = polarizer->getOutputBuffer(),
        .phasors = phasor->getOutputPhasors(),
    });

    if (config.detectorEnable) {
        BL_DEBUG("Instantiating detector module.");
        this->connect(detector, {
            .integrationSize = config.detectorIntegrationSize,
            .numberOfOutputPolarizations = config.detectorNumberOfOutputPolarizations,

            .blockSize = config.detectorBlockSize,
        }, {
            .buf = beamformer->getOutputBuffer(),
        });

        if (config.detectorTransposedATPFrevOutput) {
            BL_DEBUG("Instantiating transposer module.");
            this->connect(transposerFReversed, {
            }, {
                .buf = detector->getOutputBuffer(),
            } );
        }
        else if (config.detectorTransposedATPFOutput) {
            BL_DEBUG("Instantiating transposer module.");
            this->connect(transposer, {
            }, {
                .buf = detector->getOutputBuffer(),
            } );
        }

        if constexpr (!std::is_same<OT, F32>::value) {
            BL_DEBUG("Instantiating output cast from F32 to {}.", TypeInfo<OT>::name);
            if (config.detectorTransposedATPFrevOutput) {
                this->connect(outputCast, {
                    .blockSize = config.castBlockSize,
                }, {
                    .buf = transposerFReversed->getOutputBuffer(),
                });
            } else if (config.detectorTransposedATPFOutput) {
                this->connect(outputCast, {
                    .blockSize = config.castBlockSize,
                }, {
                    .buf = transposer->getOutputBuffer(),
                });
            } else {
                this->connect(outputCast, {
                    .blockSize = config.castBlockSize,
                }, {
                    .buf = detector->getOutputBuffer(),
                });
            }
        }
    } else {
        if (config.detectorTransposedATPFrevOutput) {
            BL_WARN("Transposition will not be enabled as the detector module is disabled.");
        }

        if constexpr (!std::is_same<OT, CF32>::value) {
            BL_DEBUG("Instantiating output cast from CF32 to {}.", TypeInfo<OT>::name);
            this->connect(complexOutputCast, {
                .blockSize = config.castBlockSize,
            }, {
                .buf = beamformer->getOutputBuffer(),
            });
        }
    }
}

template<typename IT, typename OT>
const Result ModeB<IT, OT>::transferIn(const Vector<Device::CPU, F64>& blockJulianDate,
                                   const Vector<Device::CPU, F64>& blockDut1,
                                   const Vector<Device::CPU, U64>& blockFrequencyChannelOffset,
                                   const ArrayTensor<Device::CPU, IT>& input,
                                   const cudaStream_t& stream) { 
    if (this->config.accumulateRate > 1) {
        BL_FATAL("Configured to accumulate, cannot simply `transferIn`.");
        return Result::ASSERTION_ERROR;
    }
    // Copy input to static buffers.
    BL_CHECK(Memory::Copy(this->blockJulianDate, blockJulianDate));
    BL_CHECK(Memory::Copy(this->blockDut1, blockDut1));
    BL_CHECK(Memory::Copy(this->blockFrequencyChannelOffset, blockFrequencyChannelOffset));
    BL_CHECK(Memory::Copy(this->input, input, stream));

    // Print dynamic arguments on first run.
    if (this->getCurrentComputeCount() == 0) {
        BL_DEBUG("Block Julian Date: {}", this->blockJulianDate[0]);
        BL_DEBUG("Block DUT1: {}", this->blockDut1[0]);
    }

    return Result::SUCCESS;
}

template<typename IT, typename OT>
const Result ModeB<IT, OT>::accumulate(const Vector<Device::CPU, F64>& blockJulianDate,
                                   const Vector<Device::CPU, F64>& blockDut1,
                                   const Vector<Device::CPU, U64>& blockFrequencyChannelOffset,
                                   const ArrayTensor<Device::CPU, IT>& data,
                                   const cudaStream_t& stream) { 
    // Copy input to static buffers.
    BL_CHECK(Memory::Copy(this->blockJulianDate, blockJulianDate));
    BL_CHECK(Memory::Copy(this->blockDut1, blockDut1));
    BL_CHECK(Memory::Copy(this->blockFrequencyChannelOffset, blockFrequencyChannelOffset));

    if (config.inputDimensions != data.dims()) {
        BL_FATAL("Configured for array of shape {}, cannot accumulate shape {}.", config.inputDimensions, input.dims());
        return Result::ASSERTION_ERROR;
    }

    // Accumulate AFTP buffers across the T dimension
    const auto& inputHeight = config.inputDimensions.numberOfAspects() * config.inputDimensions.numberOfFrequencyChannels();
    const auto& inputWidth = data.size_bytes() / inputHeight;

    const auto& outputPitch = this->input.size_bytes() / inputHeight;
    BL_CHECK(
        Memory::Copy2D(
            this->input,
            outputPitch, // dstStride
            outputPitch * this->getCurrentAccumulatorStep(), // dstOffset

            data,
            inputWidth,
            0,

            inputWidth,
            inputHeight, 
            stream
        )
    );

    // Print dynamic arguments on first run.
    if (this->getCurrentComputeCount() == 0) {
        BL_DEBUG("Block Julian Date: {}", this->blockJulianDate[0]);
        BL_DEBUG("Block DUT1: {}", this->blockDut1[0]);
    }

    return Result::SUCCESS;
}

template<typename IT, typename OT>
const Result ModeB<IT, OT>::accumulate(const Vector<Device::CPU, F64>& blockJulianDate,
                                   const Vector<Device::CPU, F64>& blockDut1,
                                   const Vector<Device::CPU, U64>& blockFrequencyChannelOffset,
                                   const ArrayTensor<Device::CUDA, IT>& data,
                                   const cudaStream_t& stream) { 
    // Copy input to static buffers.
    BL_CHECK(Memory::Copy(this->blockJulianDate, blockJulianDate));
    BL_CHECK(Memory::Copy(this->blockDut1, blockDut1));
    BL_CHECK(Memory::Copy(this->blockFrequencyChannelOffset, blockFrequencyChannelOffset));

    if (config.inputDimensions != data.dims()) {
        BL_FATAL("Configured for array of shape {}, cannot accumulate shape {}.", config.inputDimensions, input.dims());
        return Result::ASSERTION_ERROR;
    }

    // Accumulate AFTP buffers across the T dimension
    const auto& inputHeight = config.inputDimensions.numberOfAspects() * config.inputDimensions.numberOfFrequencyChannels();
    const auto& inputWidth = data.size_bytes() / inputHeight;

    const auto& outputPitch = inputWidth * this->config.accumulateRate;

    BL_CHECK(
        Memory::Copy2D(
            this->input,
            outputPitch, // dstStride
            inputWidth * this->getCurrentAccumulatorStep(), // dstOffset

            data,
            inputWidth,
            0,

            inputWidth,
            inputHeight, 
            stream
        )
    );

    // Print dynamic arguments on first run.
    if (this->getCurrentComputeCount() == 0) {
        BL_DEBUG("Block Julian Date: {}", this->blockJulianDate[0]);
        BL_DEBUG("Block DUT1: {}", this->blockDut1[0]);
    }

    return Result::SUCCESS;
}

template class BLADE_API ModeB<CI8, CF32>;
template class BLADE_API ModeB<CI8, CF16>;
template class BLADE_API ModeB<CI8, F32>;
template class BLADE_API ModeB<CI8, F16>;

template class BLADE_API ModeB<CF32, CF32>;
template class BLADE_API ModeB<CF32, CF16>;
template class BLADE_API ModeB<CF32, F32>;
template class BLADE_API ModeB<CF32, F16>;

}  // namespace Blade::Pipelines::ATA
