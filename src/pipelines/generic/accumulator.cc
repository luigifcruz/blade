#define BL_LOG_DOMAIN "P::ACCUMULATOR"

#include "blade/pipelines/generic/accumulator.hh"

namespace Blade::Pipelines::Generic {

template<typename ModuleType, Device Dev, typename InputType>
Accumulator<ModuleType, Dev, InputType>::Accumulator(const Config& config) 
     : Pipeline(config.accumulateRate, 1),
       config(config) {
    BL_DEBUG("Initializing CLI Accumulator Pipeline.");

    // TODO break away from implicit ArrayDimensions
    // and Frequency accumulation limitations
    this->accumulationBuffer.resize(
        config.inputDimensions * ArrayDimensions({
            .A = 1,
            .F = config.accumulateRate,
            .T = 1,
            .P = 1,
        })
    );

    BL_INFO("Step Dimensions [A, F, T, P]: {} -> {}", config.inputDimensions, "N/A");
    BL_INFO("Total Dimensions [A, F, T, P]: {} -> {}", this->accumulationBuffer.dims(), "N/A");

    BL_DEBUG("Instantiating accumulated module.");
    this->connect(moduleUnderlying, config.moduleConfig, {
        .buffer = accumulationBuffer,
    });
}

template<typename ModuleType, Device Dev, typename InputType>
const Result Accumulator<ModuleType, Dev, InputType>::accumulate(const ArrayTensor<Device::CUDA, InputType>& data,
                                        const cudaStream_t& stream) {
    const auto stepInputDims = data.dims();
    if (stepInputDims != this->config.inputDimensions) {
        BL_FATAL("Accumulate input dimensions ({}) mismatches writer step input buffer dimensions ({}).",
            stepInputDims, this->config.inputDimensions);
        return Result::ASSERTION_ERROR;
    }

    if (this->config.inputIsATPFNotAFTP && this->config.reconstituteBatchedDimensions) {
        // reconstitute ATPF from batches
        const auto accumulationDims = this->accumulationBuffer.dims();
        const U64 outputByteStrideA = this->accumulationBuffer.size_bytes() / accumulationDims.numberOfAspects();
        const U64 outputByteStrideT = outputByteStrideA / accumulationDims.numberOfTimeSamples();
        const U64 outputByteStrideP = outputByteStrideT / accumulationDims.numberOfPolarizations();
        const U64 outputByteStrideF = outputByteStrideP / accumulationDims.numberOfFrequencyChannels();

        const U64 inputByteStrideA = data.size_bytes() / stepInputDims.numberOfAspects();
        const U64 inputByteStrideT = inputByteStrideA / stepInputDims.numberOfTimeSamples();
        const U64 inputByteStrideP = inputByteStrideT / stepInputDims.numberOfPolarizations();

        const auto frequencyChannelBatchIndex = (this->getAccumulatorNumberOfSteps()-1 - this->getCurrentAccumulatorStep());
        const auto outputFrequencyChannelByteOffset = frequencyChannelBatchIndex*stepInputDims.numberOfFrequencyChannels()*outputByteStrideF;

        for(U64 a = 0; a < stepInputDims.numberOfAspects(); a++) {
            BL_CHECK(
                Memory::Copy2D(
                    this->accumulationBuffer,
                    outputByteStrideP, // dstPitch
                    a*outputByteStrideA + outputFrequencyChannelByteOffset, // dstOffset

                    data,
                    inputByteStrideP, // srcPitch
                    a*inputByteStrideA, // srcOffset

                    inputByteStrideP, // (FrequencyChannels) width
                    stepInputDims.numberOfTimeSamples()*stepInputDims.numberOfPolarizations(), // height
                    stream
                )
            );
        }
        return Result::SUCCESS;
    }
    else if (this->config.reconstituteBatchedDimensions) {
        const auto numberOfInputFrequencyChannelBatches = this->getAccumulatorNumberOfSteps();
        const auto frequencyChannelBatchIndex = this->getCurrentAccumulatorStep();
        auto accumulationBufferBatchSegment = ArrayTensor<Dev, InputType>(accumulationBuffer.data() + frequencyChannelBatchIndex*stepInputDims.size(), stepInputDims);

        const U64 numberOfTimePolarizationSamples = stepInputDims.numberOfTimeSamples()*stepInputDims.numberOfPolarizations();
        const U64 numberOfAspects = stepInputDims.numberOfAspects();
        const U64 stepNumberOfFrequencyChannels = stepInputDims.numberOfFrequencyChannels();
        const U64 totalNumberOfFrequencyChannels = stepNumberOfFrequencyChannels * numberOfInputFrequencyChannelBatches;

        BL_CHECK(
            Memory::Copy2D(
                accumulationBuffer,
                totalNumberOfFrequencyChannels*numberOfTimePolarizationSamples*sizeof(InputType), // dstPitch
                frequencyChannelBatchIndex*stepNumberOfFrequencyChannels*numberOfTimePolarizationSamples*sizeof(InputType), // dstOffset 
                data,
                stepNumberOfFrequencyChannels*numberOfTimePolarizationSamples*sizeof(InputType), // srcPitch
                0, // srcOffset
                stepNumberOfFrequencyChannels*numberOfTimePolarizationSamples*sizeof(InputType),
                numberOfAspects,
                stream
            )
        );

    }
    else {
        const auto offset = this->getCurrentAccumulatorStep() * stepInputDims.size();
        auto buffer = ArrayTensor<Dev, InputType>(accumulationBuffer.data() + offset, stepInputDims);
        BL_CHECK(Memory::Copy(buffer, data, stream));
    }

    return Result::SUCCESS;
}

template class BLADE_API Accumulator<Modules::Guppi::Writer<CF16>, Device::CPU, CF16>;
template class BLADE_API Accumulator<Modules::Guppi::Writer<CF32>, Device::CPU, CF32>;

template class BLADE_API Accumulator<Modules::Filterbank::Writer<F16>, Device::CPU, F16>;
template class BLADE_API Accumulator<Modules::Filterbank::Writer<F32>, Device::CPU, F32>;
template class BLADE_API Accumulator<Modules::Filterbank::Writer<F64>, Device::CPU, F64>;

}  // namespace Blade::Pipelines::Generic
