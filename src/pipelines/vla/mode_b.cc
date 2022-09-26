#define BL_LOG_DOMAIN "P::VLA::MODE_B"

#include "blade/pipelines/vla/mode_b.hh"

namespace Blade::Pipelines::VLA {

template<typename IT, typename OT>
ModeB<IT, OT>::ModeB(const Config& config) : config(config) {
    BL_DEBUG("Initializing VLA Pipeline Mode B.");

    BL_DEBUG("Allocating pipeline buffers.");
    BL_CHECK_THROW(this->input.resize(config.inputDimensions));
    BL_CHECK_THROW(this->phasors.resize(config.beamformerPhasors.dims()));
    BL_CHECK_THROW(Memory::Copy(this->phasors, config.beamformerPhasors));

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
		}

        BL_DEBUG("Instantiating beamformer module.");
        this->connect(beamformer, {
            .enableIncoherentBeam = config.beamformerIncoherentBeam,
            .enableIncoherentBeamSqrt = (config.detectorEnable) ? true : false,

            .blockSize = config.beamformerBlockSize,
        }, {
            .buf = channelizer->getOutputBuffer(),
            .phasors = this->phasors,
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
const Result ModeB<IT, OT>::transferIn(const ArrayTensor<Device::CPU, IT>& input,
                                   const cudaStream_t& stream) { 
    // Copy input to static buffers.
    BL_CHECK(Memory::Copy(this->input, input, stream));

    // Print dynamic arguments on first run.
    if (this->getCurrentComputeStep() == 0) {
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
