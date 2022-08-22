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
    }, {});

    // Checking file and recipe bounds.

    if (guppi->getTotalNumberOfAntennas() != bfr5->getTotalNumberOfAntennas()) {
        BL_FATAL("Number of antennas from GUPPI RAW ({}) and BFR5 ({}) files mismatch.", 
                guppi->getTotalNumberOfAntennas(), bfr5->getTotalNumberOfAntennas());
        BL_CHECK_THROW(Result::ASSERTION_ERROR);
    }

    if (guppi->getTotalNumberOfFrequencyChannels() != bfr5->getTotalNumberOfFrequencyChannels()) {
        BL_FATAL("Number of frequency channels from GUPPI RAW ({}) and BFR5 ({}) files mismatch.", 
                guppi->getTotalNumberOfFrequencyChannels(), bfr5->getTotalNumberOfFrequencyChannels());
        BL_CHECK_THROW(Result::ASSERTION_ERROR);
    }

    if (guppi->getTotalNumberOfPolarizations() != bfr5->getTotalNumberOfPolarizations()) {
        BL_FATAL("Number of polarizations from GUPPI RAW ({}) and BFR5 ({}) files mismatch.", 
                guppi->getTotalNumberOfPolarizations(), bfr5->getTotalNumberOfPolarizations());
        BL_CHECK_THROW(Result::ASSERTION_ERROR);
    }
}

template<typename OT>
const Result FileReader<OT>::run() {
    if (!guppi->keepRunning()) {
        return Result::EXHAUSTED;
    }

    BL_CHECK(this->compute());
    BL_CHECK(this->synchronize());

    return Result::SUCCESS;
}

template class BLADE_API FileReader<CI8>;

}  // namespace Blade::Pipelines::Generic