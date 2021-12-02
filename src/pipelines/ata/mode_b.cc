#include "blade/pipelines/ata/mode_b.hh"

namespace Blade::Pipelines::ATA {

ModeB::ModeB(const Config& configuration) :
    Pipeline(true, false), config(configuration) {
    if (this->setup() != Result::SUCCESS) {
        throw Result::ERROR;
    }
}

Result ModeB::run(const std::span<CI8>& in, std::span<uint8_t>& out) {
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
    #if BLADE_ATA_MODE_B_OUTPUT_NCOMPLEX_BYTES != 8
    BL_CHECK(allocateBuffer(bufferE, beamformer->getOutputSize()));
    #endif

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
    #if BLADE_ATA_MODE_B_OUTPUT_NCOMPLEX_BYTES != 8
    BL_CHECK(cast->run(bufferD, bufferE, cudaStream));
    #endif

    return Result::SUCCESS;
}

Result ModeB::loopDownload() {
    #if BLADE_ATA_MODE_B_OUTPUT_NCOMPLEX_BYTES != 8
    BL_CHECK(this->copyBuffer2D(
        output,
        BLADE_ATA_MODE_B_OUTPUT_MEMCPY2D_DPITCH,
        bufferE,
        BLADE_ATA_MODE_B_OUTPUT_MEMCPY2D_WIDTH,
        BLADE_ATA_MODE_B_OUTPUT_MEMCPY2D_WIDTH,
        bufferE.size_bytes()/BLADE_ATA_MODE_B_OUTPUT_MEMCPY2D_WIDTH,
        CopyKind::D2H));
    #else // copy directly from beamformer output
    BL_CHECK(this->copyBuffer2D(
        output,
        BLADE_ATA_MODE_B_OUTPUT_MEMCPY2D_DPITCH,
        bufferD,
        BLADE_ATA_MODE_B_OUTPUT_MEMCPY2D_WIDTH,
        BLADE_ATA_MODE_B_OUTPUT_MEMCPY2D_WIDTH,
        bufferD.size_bytes()/BLADE_ATA_MODE_B_OUTPUT_MEMCPY2D_WIDTH,
        CopyKind::D2H));
    #endif

    return Result::SUCCESS;
}

}  // namespace Blade::Pipelines::ATA
