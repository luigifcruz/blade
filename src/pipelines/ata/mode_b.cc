#include "blade/pipelines/ata/mode_b.hh"

namespace Blade::Pipelines::ATA {

ModeB::ModeB(const Config& config) : config(config) {
    this->connect(inputCast, {
        .inputSize = config.inputDims.getSize(),
        .blockSize = config.castBlockSize,
    }, {input});

    #if BLADE_ATA_MODE_B_CHANNELISER_RATE > 1
    BL_INFO("Instantiating channelizer with FFT Size {}.", config.channelizerRate);
    this->connect(channelizer, {
        .dims = config.inputDims,
        .fftSize = config.channelizerRate,
        .blockSize = config.channelizerBlockSize,
    }, {inputCast->getOutput()});

    auto dims = channelizer->getOutputDims();
    #else
    BL_INFO("Not instantiating channelizer.");
    auto dims = config.inputDims;
    #endif
    dims.NBEAMS *= config.beamformerBeams;

    this->connect(beamformer, {
        .dims = dims,
        .blockSize = config.beamformerBlockSize,
    }, {
    #if BLADE_ATA_MODE_B_CHANNELISER_RATE > 1
        channelizer->getOutput(),
    #else
        inputCast->getOutput(),
    #endif
    phasors});

    #if BLADE_ATA_MODE_B_OUTPUT_NCOMPLEX_BYTES != 8
    BL_INFO("Instantiating outputCast to {}.", "BLADE_ATA_MODE_B_OUTPUT_ELEMENT_T");
    // cast from CF32 to BLADE_ATA_MODE_B_OUTPUT_ELEMENT_T
    this->connect(outputCast, {
        .inputSize = beamformer->getOutputSize(),
        .blockSize = config.castBlockSize,
    }, {beamformer->getOutput()});
    #else
    BL_INFO("Not instantiating outputCast.");
    #endif
}

Result ModeB::run(const Vector<Device::CPU, CI8>& input,
                        Vector<Device::CPU, BLADE_ATA_MODE_B_OUTPUT_ELEMENT_T>& output) {
    BL_CHECK(this->copy(inputCast->getInput(), input));
    BL_CHECK(this->compute());
    #if BLADE_ATA_MODE_B_OUTPUT_NCOMPLEX_BYTES != 8
    // output is casted output
    BL_CHECK(this->copy2D(
        output,
        BLADE_ATA_MODE_B_OUTPUT_MEMCPY2D_DPITCH,// dpitch
        outputCast->getOutput(),                // src
        BLADE_ATA_MODE_B_OUTPUT_MEMCPY2D_WIDTH, // spitch
        BLADE_ATA_MODE_B_OUTPUT_MEMCPY2D_WIDTH, // width
        (beamformer->getOutputSize()*sizeof(BLADE_ATA_MODE_B_OUTPUT_ELEMENT_T))/BLADE_ATA_MODE_B_OUTPUT_MEMCPY2D_WIDTH
        ));
    #else
    // output is un-casted beamformer output (CF32)
    BL_CHECK(this->copy2D(
        output,
        BLADE_ATA_MODE_B_OUTPUT_MEMCPY2D_DPITCH,// dpitch
        beamformer->getOutput(),                // src
        BLADE_ATA_MODE_B_OUTPUT_MEMCPY2D_WIDTH, // spitch
        BLADE_ATA_MODE_B_OUTPUT_MEMCPY2D_WIDTH, // width
        (beamformer->getOutputSize()*sizeof(BLADE_ATA_MODE_B_OUTPUT_ELEMENT_T))/BLADE_ATA_MODE_B_OUTPUT_MEMCPY2D_WIDTH
        ));
    #endif

    return Result::SUCCESS;
}

}  // namespace Blade::Pipelines::ATA
