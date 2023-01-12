#define BL_LOG_DOMAIN "P::MODE_S"

#include "blade/pipelines/generic/mode_s.hh"

namespace Blade::Pipelines::Generic {

ModeS::ModeS(const Config& config)
     : Pipeline(1, 1),
       config(config),
       coarseFrequencyChannelOffset({1}) {
    BL_DEBUG("Initializing Pipeline Mode S.");

    BL_DEBUG("Allocating pipeline buffers.");
    BL_CHECK_THROW(this->input.resize(config.inputDimensions));
    BL_CHECK_THROW(this->prebeamformerData.resize(config.prebeamformerInputDimensions));

    BL_DEBUG("Instantiating Dedoppler module.");
    this->connect(this->dedoppler, {
        .mitigateDcSpike = config.searchMitigateDcSpike,
        .minimumDriftRate = config.searchMinimumDriftRate,
        .maximumDriftRate = config.searchMaximumDriftRate,
        .snrThreshold = config.searchSnrThreshold,
        .channelBandwidthHz = config.searchChannelBandwidthHz,
        .channelTimespanS = config.searchChannelTimespanS,
        .coarseChannelRate = config.inputCoarseChannelRate,
        .lastBeamIsIncoherent = config.inputLastBeamIsIncoherent,
    }, {
        .buf = this->input,
        .coarseFrequencyChannelOffset = this->coarseFrequencyChannelOffset,
    });

    BL_DEBUG("Instantiating HitsWriter module.");
    this->connect(this->hitsWriter, {
        .filepathPrefix = config.searchOutputFilepathStem,
        .directio = true,
        .channelBandwidthHz = config.searchChannelBandwidthHz,
        .channelTimespanS = config.searchChannelTimespanS,
    }, {
        .buffer = this->prebeamformerData,
        .hits = this->dedoppler->getOutputHits(),
    });

}

void ModeS::setFrequencyOfFirstInputChannel(F64 hz) {
    dedoppler->setFrequencyOfFirstInputChannel(hz);
}

const Result ModeS::accumulate(const ArrayTensor<Device::CUDA, F32>& data,
                               const ArrayTensor<Device::CPU, CF32>& prebeamformerData,
                               const Vector<Device::CPU, U64>& coarseFrequencyChannelOffset,
                               const cudaStream_t& stream) {
    // Accumulate ATPF in the time domain
    if (config.inputDimensions != data.dims()) {
        BL_FATAL("Configured for array of shape {}, cannot receive shape {}.", config.inputDimensions, data.dims());
        return Result::ASSERTION_ERROR;
    }
    BL_CHECK(Memory::Copy(
        this->coarseFrequencyChannelOffset,
        coarseFrequencyChannelOffset
    ));
    BL_CHECK(Memory::Copy(
        this->prebeamformerData,
        prebeamformerData
    ));
    BL_CHECK(Memory::Copy(
        this->input,
        data
    ));

    return Result::SUCCESS;
}

const Result ModeS::accumulate(const ArrayTensor<Device::CUDA, F32>& data,
                               const ArrayTensor<Device::CUDA, CF32>& prebeamformerData,
                               const Vector<Device::CPU, U64>& coarseFrequencyChannelOffset,
                               const cudaStream_t& stream) {
    // Accumulate ATPF in the time domain
    if (config.inputDimensions != data.dims()) {
        BL_FATAL("Configured for array of shape {}, cannot receive shape {}.", config.inputDimensions, data.dims());
        return Result::ASSERTION_ERROR;
    }
    BL_CHECK(Memory::Copy(
        this->coarseFrequencyChannelOffset,
        coarseFrequencyChannelOffset
    ));
    BL_CHECK(Memory::Copy(
        this->prebeamformerData,
        prebeamformerData
    ));
    BL_CHECK(Memory::Copy(
        this->input,
        data
    ));

    return Result::SUCCESS;
}

}  // namespace Blade::Pipelines::Generic
