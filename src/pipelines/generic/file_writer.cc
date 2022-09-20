#define BL_LOG_DOMAIN "P::FILE_WRITER"

#include "blade/pipelines/generic/file_writer.hh"

namespace Blade::Pipelines::Generic {

template<typename IT>
FileWriter<IT>::FileWriter(const Config& config) 
     : Accumulator(config.accumulateRate),
       config(config) {
    BL_DEBUG("Initializing CLI File Writer Pipeline.");

    BL_DEBUG("Instantiating GUPPI RAW file writer.");
    this->connect(guppi, {
        .filepath = config.outputGuppiFile,
        .directio = config.directio,

        .inputFrequencyBatches = this->getAccumulatorNumberOfSteps(),

        .blockSize = config.writerBlockSize,
    }, {
        .buffer = writerBuffer,
    });
}

template<typename IT>
const Result FileWriter<IT>::accumulate(const ArrayTensor<Device::CUDA, IT>& data,
                                        const cudaStream_t& stream) {
    const auto stepInputBufferSize = this->getStepInputBufferSize();
    if (stepInputBufferSize != data.size()) {
        BL_FATAL("Accumulate input size ({}) mismatches writer step input buffer size ({}).",
            data.size(), stepInputBufferSize);
        return Result::ASSERTION_ERROR;
    }

    const auto offset = this->getCurrentAccumulatorStep() * stepInputBufferSize;
    auto input = ArrayTensor<Device::CPU, IT>(writerBuffer.data() + offset, data.dims());
    BL_CHECK(Memory::Copy(input, data, stream));

    return Result::SUCCESS;
}

template class BLADE_API FileWriter<CF16>;
template class BLADE_API FileWriter<CF32>;

}  // namespace Blade::Pipelines::Generic
