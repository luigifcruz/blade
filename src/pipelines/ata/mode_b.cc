#include "blade/pipelines/ata/mode_b.hh"

namespace Blade::Pipelines::ATA {

ModeB::ModeB(const Config& config) : config(config) {
    this->connect(inputCast, {
        .inputSize = config.inputDims.getSize(),
        .blockSize = config.castBlockSize,
    }, {input});

    this->connect(channelizer, {
        .dims = config.inputDims,
        .fftSize = config.channelizerRate,
        .blockSize = config.channelizerBlockSize,
    }, {inputCast->getOutput()});

    auto dims = channelizer->getOutputDims();
    dims.NBEAMS *= config.beamformerBeams;

    this->connect(beamformer, {
        .dims = dims,
        .blockSize = config.beamformerBlockSize,
    }, {channelizer->getOutput(), phasors});

    this->connect(outputCast, {
        .inputSize = beamformer->getOutputSize(),
        .blockSize = config.castBlockSize,
    }, {beamformer->getOutput()});
}

Result ModeB::run(const Vector<Device::CPU, CI8>& input,
                  const Vector<Device::CPU, CF32>& phasors,
                        Vector<Device::CPU, CF16>& output) {
    BL_CHECK(this->copy(inputCast->getInput(), input));
    BL_CHECK(this->copy(beamformer->getPhasors(), phasors));
    BL_CHECK(this->compute());
    BL_CHECK(this->copy(output, outputCast->getOutput()));

    return Result::SUCCESS;
}

}  // namespace Blade::Pipelines::ATA
