#define BL_LOG_DOMAIN "P::MODE_S"

#include "blade/pipelines/generic/mode_s.hh"

namespace Blade::Pipelines::Generic {

template<HitsFormat HT>
ModeS<HT>::ModeS(const Config& config)
     : Pipeline(config.accumulateRate, 1),
       config(config),
       coarseFrequencyChannelOffset({1}),
       frequencyOfFirstChannelHz({1}),
       julianDateStart({1}) {
    BL_DEBUG("Initializing Pipeline Mode S.");

    ArrayDimensions accumulationDimensionRates = {.A=1, .F=1, .T=config.accumulateRate, .P=1};
    BL_DEBUG("Allocating pipeline buffers.");
    ArrayDimensions inputAccumulatedDimensions = accumulationDimensionRates * config.inputDimensions;
    if (config.accumulateRate > 1) {
        BL_DEBUG("Input Dimensions: {}", config.inputDimensions);
        BL_DEBUG("Accumulated Input Dimensions: {}", inputAccumulatedDimensions);
    }
    BL_CHECK_THROW(this->input.resize(inputAccumulatedDimensions));
    if (config.accumulateRate > 1) {
        BL_DEBUG("Accumulated Input Byte Size: {}", this->input.size_bytes());
    }

    ArrayDimensions prebeamformerAccumulatedDimensions = accumulationDimensionRates * config.prebeamformerInputDimensions;
    if (config.accumulateRate > 1) {
        BL_DEBUG("Input Pre-Beamformer Dimensions: {}", config.prebeamformerInputDimensions);
        BL_DEBUG("Accumulated Input Dimensions: {}", prebeamformerAccumulatedDimensions);
    }

    BL_CHECK_THROW(this->prebeamformerData.resize(prebeamformerAccumulatedDimensions));
    if (config.accumulateRate > 1) {
        BL_DEBUG("Accumulated Input Byte Size: {}", this->prebeamformerData.size_bytes());
    }

    BL_DEBUG("Instantiating Dedoppler module.");
    F64 minimumDriftRate = config.searchMinimumDriftRate;
    if (config.searchDriftRateZeroExcluded && minimumDriftRate == 0.0) {
        // set the minimum to at least the drift-rate-resolution:
        // Calculation taken from the private Dedoppler.drift_rate_resolution.
        minimumDriftRate = config.searchChannelBandwidthHz / (this->input.dims().numberOfTimeSamples() * config.searchChannelTimespanS);
        BL_INFO("Set the minimum drift rate of the dedoppler search to the search's resolution of {} Hz/s to exclude zero.", minimumDriftRate);
    }
    this->connect(this->dedoppler, {
        .mitigateDcSpike = config.searchMitigateDcSpike,
        .minimumDriftRate = minimumDriftRate,
        .maximumDriftRate = config.searchMaximumDriftRate,
        .snrThreshold = config.searchSnrThreshold,
        .frequencyOfFirstChannelHz = config.inputFrequencyOfFirstChannelHz,
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

        .produceDebugHits = config.produceDebugHits,
    }, {
        .buf = this->input,
        .coarseFrequencyChannelOffset = this->coarseFrequencyChannelOffset,
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
            .stampFrequencyMarginHz = config.produceDebugHits ? 0.0 : config.searchStampFrequencyMarginHz,
            .hitsGroupingMargin = config.produceDebugHits ? -config.inputCoarseChannelRatio : config.searchHitsGroupingMargin,
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
            .stampFrequencyMarginHz = config.produceDebugHits ? 0.0 : config.searchStampFrequencyMarginHz,
            .hitsGroupingMargin = config.produceDebugHits ? -config.inputCoarseChannelRatio : config.searchHitsGroupingMargin,
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
            + coarseFrequencyChannelOffset[0] * this->config.searchChannelBandwidthHz*this->config.inputCoarseChannelRatio;

        BL_CHECK(Memory::Copy(
            this->julianDateStart,
            julianDateStart
        ));
    }

    // Accumulate buffers across the T dimension
    if (this->getAccumulatorNumberOfSteps() == 1) {
        BL_CHECK(Memory::Copy(
            this->prebeamformerData,
            prebeamformerData
        ));
        
        BL_CHECK(Memory::Copy(
            this->input,
            data,
            stream
        ));
    }
    else {
        // Accumulate AFTP buffers across the T dimension
        U64 inputHeight = config.prebeamformerInputDimensions.numberOfAspects() * config.prebeamformerInputDimensions.numberOfFrequencyChannels();
        U64 inputWidth = prebeamformerData.size_bytes() / inputHeight;

        U64 outputPitch = inputWidth * this->getAccumulatorNumberOfSteps();

        BL_CHECK(
            Memory::Copy2D(
                this->prebeamformerData,
                outputPitch, // dstStride
                this->getCurrentAccumulatorStep() * inputWidth, // dstOffset

                prebeamformerData,
                inputWidth,
                0,

                inputWidth,
                inputHeight
            )
        );

        // Accumulate ATPF buffers across the T dimension
        inputHeight = config.inputDimensions.numberOfAspects();
        inputWidth = data.size_bytes() / inputHeight;

        outputPitch = inputWidth * this->getAccumulatorNumberOfSteps();

        BL_CHECK(
            Memory::Copy2D(
                this->input,
                outputPitch, // dstStride
                this->getCurrentAccumulatorStep() * inputWidth, // dstOffset

                data,
                inputWidth,
                0,

                inputWidth,
                inputHeight, 
                stream
            )
        );
    }

    BL_DEBUG(
        "accumulate from RAM: {}/{}\nschan: {}\nfch1: {}\njd: {}",
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
            + coarseFrequencyChannelOffset[0] * this->config.searchChannelBandwidthHz*this->config.inputCoarseChannelRatio;

        BL_CHECK(Memory::Copy(
            this->julianDateStart,
            julianDateStart
        ));
    }

    // Accumulate buffers across the T dimension
    if (this->getAccumulatorNumberOfSteps() == 1) {
        BL_CHECK(Memory::Copy(
            this->prebeamformerData,
            prebeamformerData,
            stream
        ));
        
        BL_CHECK(Memory::Copy(
            this->input,
            data,
            stream
        ));
    }
    else {
        // Accumulate AFTP buffers across the T dimension
        U64 inputHeight = config.prebeamformerInputDimensions.numberOfAspects() * config.prebeamformerInputDimensions.numberOfFrequencyChannels();
        U64 inputWidth = prebeamformerData.size_bytes() / inputHeight;

        U64 outputPitch = inputWidth * this->getAccumulatorNumberOfSteps();

        BL_CHECK(
            Memory::Copy2D(
                this->prebeamformerData,
                outputPitch, // dstStride
                this->getCurrentAccumulatorStep() * inputWidth, // dstOffset

                prebeamformerData,
                inputWidth,
                0,

                inputWidth,
                inputHeight,
                stream
            )
        );

        // Accumulate ATPF buffers across the T dimension
        inputHeight = config.inputDimensions.numberOfAspects();
        inputWidth = data.size_bytes() / inputHeight;

        outputPitch = inputWidth * this->getAccumulatorNumberOfSteps();

        BL_CHECK(
            Memory::Copy2D(
                this->input,
                outputPitch, // dstStride
                this->getCurrentAccumulatorStep() * inputWidth, // dstOffset

                data,
                inputWidth,
                0,

                inputWidth,
                inputHeight, 
                stream
            )
        );
    }

    BL_DEBUG(
        "accumulate from VRAM: {}/{}\nschan: {}\nfch1: {}\njd: {}",
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
