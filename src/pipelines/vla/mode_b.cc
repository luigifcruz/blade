#define BL_LOG_DOMAIN "P::VLA::MODE_B"

#include "blade/pipelines/vla/mode_b.hh"

namespace Blade::Pipelines::VLA {

template<typename IT, typename OT>
ModeB<IT, OT>::ModeB(const Config& config) : config(config), blockJulianDate({1}), blockDut1({1}) {
    BL_DEBUG("Initializing VLA Pipeline Mode B.");

    BL_DEBUG("Allocating pipeline buffers.");
    BL_CHECK_THROW(this->input.resize(config.inputDimensions));

    if constexpr (!std::is_same<IT, CF32>::value) {
        BL_DEBUG("Instantiating input cast from {} to CF32.", TypeInfo<IT>::name);
        this->connect(inputCast, {
            .blockSize = config.castBlockSize,
        }, {
            .buf = this->input,
        });

        BL_DEBUG("Instantiating pre-beamformer channelizer with rate {}.",
            config.preBeamformerChannelizerRate);
        this->connect(channelizer, {
            .rate = config.preBeamformerChannelizerRate,

            .blockSize = config.channelizerBlockSize,
        }, {
            .buf = this->inputCast->getOutputBuffer(),
        });
    } else {
        BL_DEBUG("No need to instantiate input cast to CF32.");

        BL_DEBUG("Instantiating pre-beamformer channelizer with rate {}.",
            config.preBeamformerChannelizerRate);
        this->connect(channelizer, {
            .rate = config.preBeamformerChannelizerRate,

            .blockSize = config.channelizerBlockSize,
        }, {
            .buf = this->input,
        });
    }

    
    BL_DEBUG("Instantiating phasor module.");
    this->connect(phasor, {
        .numberOfBeams = config.phasorBeamAntennaDelays.dims().numberOfBeams(),
        .numberOfAntennas = channelizer->getOutputBuffer().dims().numberOfAspects(),
        .numberOfFrequencyChannels = channelizer->getOutputBuffer().dims().numberOfFrequencyChannels(),
        .numberOfPolarizations = channelizer->getOutputBuffer().dims().numberOfPolarizations(),

        .channelZeroFrequencyHz = config.phasorChannelZeroFrequencyHz,
        .channelBandwidthHz = config.phasorChannelBandwidthHz,
        .frequencyStartIndex = config.phasorFrequencyStartIndex,

        .antennaCoefficients = ArrayTensor<Device::CPU, CF64>(config.phasorAntennaCoefficients),
        .beamAntennaDelays = PhasorTensor<Device::CPU, F64>(config.phasorBeamAntennaDelays),
        .delayTimes = Vector<Device::CPU, F64>(config.phasorDelayTimes),

        .preBeamformerChannelizerRate = config.preBeamformerChannelizerRate,

        .blockSize = config.phasorBlockSize,
    }, {
        .blockJulianDate = this->blockJulianDate,
        .blockDut1 = this->blockDut1,
    });

    BL_DEBUG("Instantiating beamformer module.");
    this->connect(beamformer, {
        .enableIncoherentBeam = config.beamformerIncoherentBeam,
        .enableIncoherentBeamSqrt = (config.detectorEnable) ? true : false,

        .blockSize = config.beamformerBlockSize,
    }, {
        .buf = channelizer->getOutputBuffer(),
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

        if constexpr (!std::is_same<OT, F32>::value) {
            BL_DEBUG("Instantiating output cast from F32 to {}.", TypeInfo<OT>::name);
            this->connect(outputCast, {
                .blockSize = config.castBlockSize,
            }, {
                .buf = detector->getOutputBuffer(),
            });
        }
    } else {
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
                                       const ArrayTensor<Device::CPU, IT>& input,
                                       const cudaStream_t& stream) { 
    // Copy input to static buffers.
    BL_CHECK(Memory::Copy(this->blockJulianDate, blockJulianDate));
    BL_CHECK(Memory::Copy(this->blockDut1, blockDut1));
    BL_CHECK(Memory::Copy(this->input, input, stream));

    // Print dynamic arguments on first run.
    if (this->getCurrentComputeStep() == 0) {
        BL_DEBUG("Block Julian Date: {}", this->blockJulianDate[0]);
        BL_DEBUG("Block DUT1: {}", this->blockDut1[0]);
    }

    return Result::SUCCESS;
}

template class BLADE_API ModeB<CI8, CF32>;
template class BLADE_API ModeB<CF32, CF32>;
template class BLADE_API ModeB<CI8, CF16>;
template class BLADE_API ModeB<CF32, CF16>;
template class BLADE_API ModeB<CI8, F32>;
template class BLADE_API ModeB<CF32, F32>;
template class BLADE_API ModeB<CI8, F16>;
template class BLADE_API ModeB<CF32, F16>;

}  // namespace Blade::Pipelines::VLA
