#include "blade/pipelines/ata/mode_b.hh"

namespace Blade::Pipelines::ATA {

template<typename OT>
ModeB<OT>::ModeB(const Config& config) : config(config) {
    if ((config.outputMemPad % sizeof(OT)) != 0) {
        BL_FATAL("The outputMemPad must be a multiple of the output type bytes.")
        BL_CHECK_THROW(Result::ASSERTION_ERROR);
    }

    outputMemPitch = config.outputMemPad + config.outputMemWidth;

    this->connect(inputCast, {
        .inputSize = config.inputDims.getSize(),
        .blockSize = config.castBlockSize,
    }, {
        .buf = input,
    });

    if (config.channelizerRate > 1) {
        BL_DEBUG("Instantiating channelizer with FFT Size {}.", config.channelizerRate);

        this->connect(channelizer, {
            .dims = config.inputDims,
            .fftSize = config.channelizerRate,
            .blockSize = config.channelizerBlockSize,
        }, {
            .buf = inputCast->getOutput(),
        });

        auto dims = channelizer->getOutputDims();
        dims.NBEAMS *= config.beamformerBeams;

        BL_DEBUG("Instantiating beamformer module.");

        this->connect(beamformer, {
            .dims = dims,
            .blockSize = config.beamformerBlockSize,
        }, {
            .buf = channelizer->getOutput(),
            .phasors = phasors,
        });
    } else {
        BL_DEBUG("Instantiating beamformer module.");

        auto dims = config.inputDims;
        dims.NBEAMS *= config.beamformerBeams;

        this->connect(beamformer, {
            .dims = dims,
            .blockSize = config.beamformerBlockSize,
        }, {
            .buf = inputCast->getOutput(),
            .phasors = phasors,
        });
    }

    if constexpr (!std::is_same<OT, CF32>::value) {
        BL_DEBUG("Instantiating output cast from CF32 to {}.", typeid(OT).name());

        // Cast from CF32 to output type.
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
                        Vector<Device::CPU, OT>& output) {
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

template class ModeB<CF16>;
template class ModeB<CF32>;

}  // namespace Blade::Pipelines::ATA
