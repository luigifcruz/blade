#define BL_LOG_DOMAIN "P::FILE_READER"

#include "blade/pipelines/generic/file_reader.hh"

namespace Blade::Pipelines::Generic {

template<typename OT>
FileReader<OT>::FileReader(const Config& config) : config(config) {
    BL_DEBUG("Initializing CLI File Reader Pipeline.");

    BL_DEBUG("Instantiating GUPPI RAW file reader.");
    this->connect(guppi, {
        .filepath = config.inputGuppiFile,
        .stepNumberOfTimeSamples = config.stepNumberOfTimeSamples, 
        .requiredMultipleOfTimeSamplesSteps = config.requiredMultipleOfTimeSamplesSteps,
        .stepNumberOfFrequencyChannels = config.stepNumberOfFrequencyChannels,
        .numberOfTimeSampleStepsBeforeFrequencyChannelStep = config.numberOfTimeSampleStepsBeforeFrequencyChannelStep,
        .numberOfFilesLimit = config.numberOfGuppiFilesLimit,
    }, {});

    BL_DEBUG("Instantiating BFR5 file reader.");
    this->connect(bfr5, {
        .filepath = config.inputBfr5File,
    }, {});

    // Checking file and recipe bounds.
    const auto bfr5Dims = bfr5->getDims();
    const auto guppiDims = guppi->getTotalOutputBufferDims();

    if (guppiDims.numberOfAspects() != bfr5Dims.numberOfAntennas()) {
        BL_FATAL("Number of antennas from GUPPI RAW ({}) and BFR5 ({}) files mismatch.", 
                guppiDims.numberOfAspects(), bfr5Dims.numberOfAntennas());
        BL_CHECK_THROW(Result::ASSERTION_ERROR);
    }

    if (guppi->getChannelStartIndex() + guppiDims.numberOfFrequencyChannels() > bfr5Dims.numberOfFrequencyChannels()) {
        BL_FATAL("Number of frequency channels from GUPPI RAW ({} + {}) and BFR5 ({}) files mismatch.", 
                guppi->getChannelStartIndex(), guppiDims.numberOfFrequencyChannels(), bfr5Dims.numberOfFrequencyChannels());
        BL_CHECK_THROW(Result::ASSERTION_ERROR);
    }

    if (guppiDims.numberOfPolarizations() != bfr5Dims.numberOfPolarizations()) {
        BL_FATAL("Number of polarizations from GUPPI RAW ({}) and BFR5 ({}) files mismatch.", 
                guppiDims.numberOfPolarizations(), bfr5Dims.numberOfPolarizations());
        BL_CHECK_THROW(Result::ASSERTION_ERROR);
    }
}

template class BLADE_API FileReader<CI8>;

}  // namespace Blade::Pipelines::Generic
