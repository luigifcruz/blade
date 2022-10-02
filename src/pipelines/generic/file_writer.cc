#define BL_LOG_DOMAIN "P::FILE_WRITER"

#include "blade/pipelines/generic/file_writer.hh"

namespace Blade::Pipelines::Generic {

template<typename WT, typename IT>
FileWriter<WT, IT>::FileWriter(const Config& config) 
     : Accumulator(config.accumulateRate),
       config(config) {
    BL_DEBUG("Initializing CLI File Writer Pipeline.");

    this->writerBuffer.resize(ArrayTensorDimensions({
        .A = config.inputDimensions.numberOfAspects(),
        .F = config.inputDimensions.numberOfFrequencyChannels() * config.accumulateRate,
        .T = config.inputDimensions.numberOfTimeSamples(),
        .P = config.inputDimensions.numberOfPolarizations(),
    }));

    BL_INFO("Step Dimensions [A, F, T, P]: {} -> {}", config.inputDimensions, "N/A");
    BL_INFO("Total Dimensions [A, F, T, P]: {} -> {}", this->writerBuffer.dims(), "N/A");

    BL_DEBUG("Instantiating file writer.");
    this->connect(writer, config.writerConfig, {
        .buffer = writerBuffer,
    });
}

template<typename WT, typename IT>
const Result FileWriter<WT, IT>::accumulate(const ArrayTensor<Device::CUDA, IT>& data,
                                        const cudaStream_t& stream) {
    const auto stepInputDims = data.dims();
    if (stepInputDims != this->config.inputDimensions) {
        BL_FATAL("Accumulate input dimensions ({}) mismatches writer step input buffer dimensions ({}).",
            stepInputDims, this->config.inputDimensions);
        return Result::ASSERTION_ERROR;
    }

    const auto offset = this->getCurrentAccumulatorStep() * stepInputDims.size();
    auto input = ArrayTensor<Device::CPU, IT>(writerBuffer.data() + offset, stepInputDims);
    if (this->config.transposeBTPF) {
        // from A F T P
        // to   A T P F (F is reversed)
        const U64 numberOfTimePolarizationSamples = stepInputDims.numberOfTimeSamples()*stepInputDims.numberOfPolarizations();
        const U64 numberOfFrequencyChannels = stepInputDims.numberOfFrequencyChannels();
        const U64 numberOfAspects = stepInputDims.numberOfAspects();
        
        for(U64 a = 0; a < numberOfAspects; a++) {
            for(U64 f = numberOfFrequencyChannels; f-- > 0; ) {
                const U64 aspectChannelFactor = (a*numberOfFrequencyChannels + f);
                BL_CHECK(
                    Memory::Copy2D(
                        input,
                        1, // dstPitch
                        aspectChannelFactor*numberOfTimePolarizationSamples*sizeof(IT), // dstOffset 
                        data,
                        numberOfTimePolarizationSamples, // srcPitch
                        aspectChannelFactor*numberOfTimePolarizationSamples*sizeof(IT), // srcOffset
                        sizeof(IT),
                        numberOfTimePolarizationSamples,
                        stream
                    )
                );
            }
        }
    }
    else {
        BL_CHECK(Memory::Copy(input, data, stream));
    }

    return Result::SUCCESS;
}

template class BLADE_API FileWriter<Modules::Guppi::Writer<CF16>, CF16>;
template class BLADE_API FileWriter<Modules::Guppi::Writer<CF32>, CF32>;

template class BLADE_API FileWriter<Modules::Filterbank::Writer<F16>, F16>;
template class BLADE_API FileWriter<Modules::Filterbank::Writer<F32>, F32>;
template class BLADE_API FileWriter<Modules::Filterbank::Writer<F64>, F64>;

}  // namespace Blade::Pipelines::Generic
