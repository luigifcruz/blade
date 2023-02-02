#define BL_LOG_DOMAIN "P::MODE_S"

#include "blade/pipelines/generic/mode_s.hh"

namespace Blade::Pipelines::Generic {

template<HitsFormat HT>
ModeS<HT>::ModeS(const Config& config)
     : Pipeline(1, 1),
       config(config),
       coarseFrequencyChannelOffset({1}),
       frequencyOfFirstChannelHz({1}),
       julianDateStart({1}) {
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
        .searchIncoherentBeam = config.searchIncoherentBeam,

        // hits writer requirements -_-
        .filepathPrefix = config.searchOutputFilepathStem,
        .telescopeId = config.inputTelescopeId,
        .sourceName = config.inputSourceName,
        .observationIdentifier = config.inputObservationIdentifier,
        .phaseCenter = config.inputPhaseCenter,
        .aspectNames = config.beamNames,
        .aspectCoordinates = config.beamCoordinates,
        .totalNumberOfTimeSamples = config.inputTotalNumberOfTimeSamples,
        .totalNumberOfFrequencyChannels = config.inputTotalNumberOfFrequencyChannels,
    }, {
        .buf = this->input,
        .coarseFrequencyChannelOffset = this->coarseFrequencyChannelOffset,
        .frequencyOfFirstChannel = this->frequencyOfFirstChannelHz,
        .julianDate = this->julianDateStart,
    });

    if (HT == HitsFormat::GUPPI_RAW) {
        BL_DEBUG("Instantiating HitsRawWriter module.");
        this->connect(this->hitsRawWriter, {
            .filepathPrefix = config.searchOutputFilepathStem,
            .directio = true,
            .telescopeId = config.inputTelescopeId,
            .sourceName = config.inputSourceName,
            .observationIdentifier = config.inputObservationIdentifier,
            .phaseCenter = config.inputPhaseCenter,
            .coarseStartChannelIndex = config.inputCoarseStartChannelIndex,
            .coarseChannelRatio = config.inputCoarseChannelRatio,
            .channelBandwidthHz = config.searchChannelBandwidthHz,
            .channelTimespanS = config.searchChannelTimespanS,
        }, {
            .buffer = this->prebeamformerData,
            .hits = this->dedoppler->getOutputHits(),
            .frequencyOfFirstChannelHz = this->frequencyOfFirstChannelHz,
            .julianDateStart = this->julianDateStart,
        });
    }
    else if (HT == HitsFormat::SETICORE_STAMP) {
        BL_DEBUG("Instantiating HitsStampWriter module.");
        this->connect(this->hitsStampWriter, {
            .filepathPrefix = config.searchOutputFilepathStem,
            .telescopeId = config.inputTelescopeId,
            .sourceName = config.inputSourceName,
            .observationIdentifier = config.inputObservationIdentifier,
            .phaseCenter = config.inputPhaseCenter,
            .coarseStartChannelIndex = config.inputCoarseStartChannelIndex,
            .coarseChannelRatio = config.inputCoarseChannelRatio,
            .channelBandwidthHz = config.searchChannelBandwidthHz,
            .channelTimespanS = config.searchChannelTimespanS,
        }, {
            .buffer = this->prebeamformerData,
            .hits = this->dedoppler->getOutputHits(),
            .frequencyOfFirstChannelHz = this->frequencyOfFirstChannelHz,
            .julianDateStart = this->julianDateStart,
        });
    }
}

template<HitsFormat HT>
const Result ModeS<HT>::accumulate(const ArrayTensor<Device::CUDA, F32>& data,
                               const ArrayTensor<Device::CPU, CF32>& prebeamformerData,
                               const Vector<Device::CPU, U64>& coarseFrequencyChannelOffset,
                               const Vector<Device::CPU, F64>& julianDateStart,
                               const cudaStream_t& stream) {
    // Accumulate ATPF in the time domain
    if (config.inputDimensions != data.dims()) {
        BL_FATAL("Configured for array of shape {}, cannot receive shape {}.", config.inputDimensions, data.dims());
        return Result::ASSERTION_ERROR;
    }
    if (this->getCurrentAccumulatorStep() == 0) {
        BL_CHECK(Memory::Copy(
            this->coarseFrequencyChannelOffset,
            coarseFrequencyChannelOffset
        ));
        this->frequencyOfFirstChannelHz[0] =
            this->config.inputFrequencyOfFirstChannelHz
            + coarseFrequencyChannelOffset[0] * this->config.searchChannelBandwidthHz;

        BL_CHECK(Memory::Copy(
            this->julianDateStart,
            julianDateStart
        ));
    }
    BL_CHECK(Memory::Copy(
        this->prebeamformerData,
        prebeamformerData
    ));
    BL_CHECK(Memory::Copy(
        this->input,
        data
    ));

    BL_DEBUG(
        "accumulate from CPU: {}/{}\nschan: {}\nfch1: {}\njd: {}",
        this->getCurrentAccumulatorStep(),
        this->getAccumulatorNumberOfSteps(),
        coarseFrequencyChannelOffset[0],
        frequencyOfFirstChannelHz[0],
        julianDateStart[0]
    );
    return Result::SUCCESS;
}

template<HitsFormat HT>
const Result ModeS<HT>::accumulate(const ArrayTensor<Device::CUDA, F32>& data,
                               const ArrayTensor<Device::CUDA, CF32>& prebeamformerData,
                               const Vector<Device::CPU, U64>& coarseFrequencyChannelOffset,
                               const Vector<Device::CPU, F64>& julianDateStart,
                               const cudaStream_t& stream) {
    // Accumulate ATPF in the time domain
    if (config.inputDimensions != data.dims()) {
        BL_FATAL("Configured for array of shape {}, cannot receive shape {}.", config.inputDimensions, data.dims());
        return Result::ASSERTION_ERROR;
    }
    if (this->getCurrentAccumulatorStep() == 0) {
        BL_CHECK(Memory::Copy(
            this->coarseFrequencyChannelOffset,
            coarseFrequencyChannelOffset
        ));
        this->frequencyOfFirstChannelHz[0] =
            this->config.inputFrequencyOfFirstChannelHz
            + coarseFrequencyChannelOffset[0] * this->config.searchChannelBandwidthHz;

        BL_CHECK(Memory::Copy(
            this->julianDateStart,
            julianDateStart
        ));
    }
    BL_CHECK(Memory::Copy(
        this->prebeamformerData,
        prebeamformerData
    ));
    BL_CHECK(Memory::Copy(
        this->input,
        data
    ));

    BL_DEBUG(
        "accumulate from CUDA: {}/{}\nschan: {}\nfch1: {}\njd: {}",
        this->getCurrentAccumulatorStep(),
        this->getAccumulatorNumberOfSteps(),
        coarseFrequencyChannelOffset[0],
        frequencyOfFirstChannelHz[0],
        julianDateStart[0]
    );
    return Result::SUCCESS;
}

template class BLADE_API ModeS<HitsFormat::GUPPI_RAW>;
template class BLADE_API ModeS<HitsFormat::SETICORE_STAMP>;

}  // namespace Blade::Pipelines::Generic
