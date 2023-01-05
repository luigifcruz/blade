#define BL_LOG_DOMAIN "P::MODE_S"

#include "blade/pipelines/generic/mode_s.hh"

namespace Blade::Pipelines::Generic {

ModeS::ModeS(const Config& config)
     : Pipeline(config.accumulateRate, 1),
       config(config) {
    BL_DEBUG("Initializing Pipeline Mode S.");

    BL_DEBUG("Allocating pipeline buffers.");
    const auto accumulationFactor = ArrayDimensions{.A=1, .F=1, .T=config.accumulateRate, .P=1};
    BL_CHECK_THROW(this->input.resize(config.inputDimensions * accumulationFactor));

    BL_DEBUG("Instantiating dedoppler module.");
    this->connect(this->dedoppler, {
        .mitigateDcSpike = config.searchMitigateDcSpike,
        .minimumDriftRate = config.searchMinimumDriftRate,
        .maximumDriftRate = config.searchMaximumDriftRate,
        .snrThreshold = config.searchSnrThreshold,
        .channelBandwidthHz = config.searchChannelBandwidthHz,
        .channelTimespanS = config.searchChannelTimespanS,
    }, {
        .buf = this->input,
    });
}

const Result ModeS::accumulate(const ArrayTensor<Device::CUDA, F32>& data,
                               const cudaStream_t& stream) {
    // Accumulate ATPF in the time domain
    if (config.inputDimensions != data.dims()) {
        BL_FATAL("Configured for array of shape {}, cannot accumulate shape {}.", config.inputDimensions, data.dims());
        return Result::ASSERTION_ERROR;
    }

    const auto& inputHeight = config.inputDimensions.numberOfAspects();
    const auto& inputWidth = data.size_bytes() / inputHeight;

    const auto& outputPitch = this->input.size_bytes() / inputHeight;

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

    return Result::SUCCESS;
}

}  // namespace Blade::Pipelines::Generic
