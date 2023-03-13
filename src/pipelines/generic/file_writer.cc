#define BL_LOG_DOMAIN "P::FILE_WRITER"

#include "blade/pipelines/generic/file_writer.hh"

namespace Blade::Pipelines::Generic {

template<typename IT>
FileWriter<IT>::FileWriter(const Config& config) 
     : Pipeline(config.accumulateRate),
       config(config) {
    BL_DEBUG("Initializing CLI File Writer Pipeline.");

    this->writerBuffer = ArrayTensor<Device::CUDA, IT>({
        config.inputShape.numberOfAspects(),
        config.inputShape.numberOfFrequencyChannels() * config.accumulateRate,
        config.inputShape.numberOfTimeSamples(),
        config.inputShape.numberOfPolarizations(),
    });

    BL_INFO("Step Shape [A, F, T, P]: {} -> {}", config.inputShape, "N/A");
    BL_INFO("Total Shape [A, F, T, P]: {} -> {}", this->writerBuffer.shape(), "N/A");

    BL_DEBUG("Instantiating GUPPI RAW file writer.");
    this->connect(guppi, {
        .filepath = config.outputGuppiFile,
        .directio = config.directio,

        .inputFrequencyBatches = config.accumulateRate,

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
    auto input = ArrayTensor<Device::CPU, IT>(writerBuffer.data() + offset, data.shape());
    BL_CHECK(Memory::Copy(input, data, stream));

    return Result::SUCCESS;
}

template class BLADE_API FileWriter<CF16>;
template class BLADE_API FileWriter<CF32>;

}  // namespace Blade::Pipelines::Generic
