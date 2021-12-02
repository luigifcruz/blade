#include "blade/pipelines/ata/mode_b.hh"

namespace Blade::Pipelines::ATA {

ModeB::ModeB(const Config& configuration) :
    Pipeline(true, false), config(configuration) {
    if (this->setup() != Result::SUCCESS) {
        throw Result::ERROR;
    }
}

Result ModeB::run(const std::span<CI8>& in, std::span<BLADE_ATA_MODE_B_OUTPUT_ELEMENT_T>& out) {
    this->input = in;
    this->output = out;
    return this->loop();
}

Result ModeB::setupModules() {
    BL_INFO("Initializing kernels.");

    cast = Factory<Modules::Cast>({
        .blockSize = config.castBlockSize,
    });

    channelizer = Factory<Modules::Channelizer>({
        .dims = config.inputDims,
        .fftSize = config.channelizerRate,
        .blockSize = config.channelizerBlockSize,
    });

    auto dims = channelizer->getOutputDims();
    dims.NBEAMS *= config.beamformerBeams;

    beamformer = Factory<Modules::Beamformer::ATA>({
        .dims = dims,
        .blockSize = config.beamformerBlockSize,
    });

    return Result::SUCCESS;
}

Result ModeB::setupMemory() {
    BL_INFO("Allocating resources.");

    BL_CHECK(allocateBuffer(phasors, beamformer->getPhasorsSize()));
    BL_CHECK(allocateBuffer(bufferA, channelizer->getBufferSize()));
    BL_CHECK(allocateBuffer(bufferB, channelizer->getBufferSize()));
    BL_CHECK(allocateBuffer(bufferC, channelizer->getBufferSize()));
    BL_CHECK(allocateBuffer(bufferD, beamformer->getOutputSize()));
    BL_CHECK(allocateBuffer(bufferE, beamformer->getOutputSize()));

    return Result::SUCCESS;
}

Result ModeB::loopUpload() {
    BL_CHECK(this->copyBuffer(bufferA, input, CopyKind::H2D));

    return Result::SUCCESS;
}

Result ModeB::loopProcess(cudaStream_t& cudaStream) {
    BL_CHECK(cast->run(bufferA, bufferB, cudaStream));
    BL_CHECK(channelizer->run(bufferB, bufferC, cudaStream));
    BL_CHECK(beamformer->run(bufferC, phasors, bufferD, cudaStream));
    BL_CHECK(cast->run(bufferD, bufferE, cudaStream));

    return Result::SUCCESS;
}

Result ModeB::loopDownload() {
    BL_CHECK(this->copyBuffer(output, bufferE, CopyKind::D2H));

    return Result::SUCCESS;
}

}  // namespace Blade::Pipelines::ATA
