#include "blade/pipelines/generic/file_writer.hh"

namespace Blade::Pipelines::Generic {

template<typename IT>
FileWriter<IT>::FileWriter(const Config& config) : config(config) {
    BL_DEBUG("Initializing CLI File Writer Pipeline.");

    BL_DEBUG("Instantiating GUPPI RAW file writer.");
    this->connect(guppi, {
        .filepath = config.outputGuppiFile,
        .directio = config.directio,

        .stepNumberOfBeams = config.stepNumberOfBeams,
        .stepNumberOfAntennas = config.stepNumberOfAntennas,
        .stepNumberOfFrequencyChannels = config.stepNumberOfFrequencyChannels,
        .stepNumberOfTimeSamples = config.stepNumberOfTimeSamples,
        .stepNumberOfPolarizations = config.stepNumberOfPolarizations,

        .totalNumberOfFrequencyChannels = config.totalNumberOfFrequencyChannels,

        .blockSize = config.writerBlockSize,
    }, {
        .totalBuffer = writerBuffer,
    });
}

template<typename IT>
Result FileWriter<IT>::run() {
    BL_CHECK(this->compute());
    BL_CHECK(this->synchronize());

    return Result::SUCCESS;
}

template class BLADE_API FileWriter<CF16>;
template class BLADE_API FileWriter<CF32>;

}  // namespace Blade::Pipelines::Generic
