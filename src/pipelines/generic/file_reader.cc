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
        .stepNumberOfFrequencyChannels = config.stepNumberOfFrequencyChannels,
    }, {});

    BL_DEBUG("Instantiating BFR5 file reader.");
    this->connect(bfr5, {
        .filepath = config.inputBfr5File,
        .channelizerRate = config.channelizerRate,
    }, {});

    // Checking file and recipe bounds.
    const auto bfr5Dims = bfr5->getTotalDims();
    const auto guppiDims = guppi->getTotalOutputBufferDims();

    if (guppiDims.numberOfAspects() != bfr5Dims.numberOfAspects()) {
        BL_FATAL("Number of aspects from GUPPI RAW ({}) and BFR5 ({}) files mismatch.", 
                guppiDims.numberOfAspects(), bfr5Dims.numberOfAspects());
        BL_CHECK_THROW(Result::ASSERTION_ERROR);
    }

    if (guppiDims.numberOfFrequencyChannels() != bfr5Dims.numberOfFrequencyChannels()) {
        BL_FATAL("Number of frequency channels from GUPPI RAW ({}) and BFR5 ({}) files mismatch.", 
                guppiDims.numberOfFrequencyChannels(), bfr5Dims.numberOfFrequencyChannels());
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
