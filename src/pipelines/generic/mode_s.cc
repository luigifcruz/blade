#define BL_LOG_DOMAIN "P::MODE_S"

#include "blade/pipelines/generic/mode_s.hh"

namespace Blade::Pipelines::Generic {

template<HitsFormat HT>
ModeS<HT>::ModeS(const Config& config)
     : Pipeline(1, 1),
       config(config),
       coarseFrequencyChannelOffset({1}),
       frequencyOfFirstInputChannelHz({1}) {
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
        .coarseChannelRate = config.inputCoarseChannelRatio,
        .lastBeamIsIncoherent = config.inputLastBeamIsIncoherent,
    }, {
        .buf = this->input,
        .coarseFrequencyChannelOffset = this->coarseFrequencyChannelOffset,
    });

    if (HT == HitsFormat::GUPPI_RAW) {
        BL_DEBUG("Instantiating HitsRawWriter module.");
        this->connect(this->hitsRawWriter, {
            .filepathPrefix = config.searchOutputFilepathStem,
            .directio = true,
            .telescopeId = config.inputTelescopeId,
            .sourceName = config.inputSourceName,
            .observationIdentifier = config.inputObservationIdentifier,
            .rightAscension = config.inputRightAscension,
            .declination = config.inputDeclination,
            .coarseStartChannelIndex = config.inputCoarseStartChannelIndex,
            .coarseChannelRatio = config.inputCoarseChannelRatio,
            .channelBandwidthHz = config.searchChannelBandwidthHz,
            .channelTimespanS = config.searchChannelTimespanS,
            .julianDateStart = config.inputJulianDateStart,
        }, {
            .buffer = this->prebeamformerData,
            .hits = this->dedoppler->getOutputHits(),
            .frequencyOfFirstInputChannelHz = this->frequencyOfFirstInputChannelHz,
        });
    }
    else if (HT == HitsFormat::SETICORE_STAMP) {
        BL_DEBUG("Instantiating HitsStampWriter module.");
        this->connect(this->hitsStampWriter, {
            .filepathPrefix = config.searchOutputFilepathStem,
            .telescopeId = config.inputTelescopeId,
            .sourceName = config.inputSourceName,
            .observationIdentifier = config.inputObservationIdentifier,
            .rightAscension = config.inputRightAscension,
            .declination = config.inputDeclination,
            .coarseStartChannelIndex = config.inputCoarseStartChannelIndex,
            .coarseChannelRatio = config.inputCoarseChannelRatio,
            .channelBandwidthHz = config.searchChannelBandwidthHz,
            .channelTimespanS = config.searchChannelTimespanS,
            .julianDateStart = config.inputJulianDateStart,
        }, {
            .buffer = this->prebeamformerData,
            .hits = this->dedoppler->getOutputHits(),
            .frequencyOfFirstInputChannelHz = this->frequencyOfFirstInputChannelHz,
        });
    }
}

template<HitsFormat HT>
void ModeS<HT>::setFrequencyOfFirstInputChannel(F64 hz) {
    dedoppler->setFrequencyOfFirstInputChannel(hz);
    this->frequencyOfFirstInputChannelHz[0] = hz;
}

template<HitsFormat HT>
const Result ModeS<HT>::accumulate(const ArrayTensor<Device::CUDA, F32>& data,
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

template<HitsFormat HT>
const Result ModeS<HT>::accumulate(const ArrayTensor<Device::CUDA, F32>& data,
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

template class BLADE_API ModeS<HitsFormat::GUPPI_RAW>;
template class BLADE_API ModeS<HitsFormat::SETICORE_STAMP>;

}  // namespace Blade::Pipelines::Generic
