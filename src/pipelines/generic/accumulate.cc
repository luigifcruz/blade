#define BL_LOG_DOMAIN "P::ACCUMULATE"

#include "blade/pipelines/generic/accumulate.hh"

namespace Blade::Pipelines::Generic {

template<typename MT, Device DT, typename IT>
Accumulate<MT, DT, IT>::Accumulate(const Config& config) 
     : Accumulator(config.accumulateRate),
       config(config) {
    BL_DEBUG("Initializing CLI Accumulate Pipeline.");

    this->accumulationBuffer.resize(ArrayTensorDimensions({
        .A = config.inputDimensions.numberOfAspects(),
        .F = config.inputDimensions.numberOfFrequencyChannels() * config.accumulateRate,
        .T = config.inputDimensions.numberOfTimeSamples(),
        .P = config.inputDimensions.numberOfPolarizations(),
    }));

    BL_INFO("Step Dimensions [A, F, T, P]: {} -> {}", config.inputDimensions, "N/A");
    BL_INFO("Total Dimensions [A, F, T, P]: {} -> {}", this->accumulationBuffer.dims(), "N/A");

    BL_DEBUG("Instantiating accumulated module.");
    this->connect(moduleUnderlying, config.moduleConfig, {
        .buffer = accumulationBuffer,
    });
}

template<typename MT, Device DT, typename IT>
const Result Accumulate<MT, DT, IT>::accumulate(const ArrayTensor<Device::CUDA, IT>& data,
                                        const cudaStream_t& stream) {
    const auto stepInputDims = data.dims();
    if (stepInputDims != this->config.inputDimensions) {
        BL_FATAL("Accumulate input dimensions ({}) mismatches writer step input buffer dimensions ({}).",
            stepInputDims, this->config.inputDimensions);
        return Result::ASSERTION_ERROR;
    }

    if (this->config.transposeBTPF) {
        // from A F T P
        // to   A T P F (F is reversed)

        // reverse the batches too seeing as they are an extension of the F dimension
        const auto offset = (this->getAccumulatorNumberOfSteps()-1 - this->getCurrentAccumulatorStep()) * stepInputDims.size();
        auto buffer = ArrayTensor<DT, IT>(accumulationBuffer.data() + offset, stepInputDims);

        const U64 numberOfTimePolarizationSamples = stepInputDims.numberOfTimeSamples()*stepInputDims.numberOfPolarizations();
        const U64 numberOfFrequencyChannels = stepInputDims.numberOfFrequencyChannels();
        const U64 numberOfAspects = stepInputDims.numberOfAspects();
        
        for(U64 a = 0; a < numberOfAspects; a++) {
            for(U64 f = 0; f < numberOfFrequencyChannels; f++) {
                const U64 aspectChannelSourceFactor = (a*numberOfFrequencyChannels + f)*numberOfTimePolarizationSamples;
                const U64 aspectChannelDestinationFactor = a*numberOfTimePolarizationSamples + (numberOfFrequencyChannels-1 - f);
                BL_CHECK(
                    Memory::Copy2D(
                        buffer,
                        numberOfFrequencyChannels*sizeof(IT), // dstPitch
                        aspectChannelDestinationFactor*sizeof(IT), // dstOffset 
                        data,
                        1*sizeof(IT), // srcPitch
                        aspectChannelSourceFactor*sizeof(IT), // srcOffset
                        sizeof(IT),
                        numberOfTimePolarizationSamples,
                        stream
                    )
                );
            }
        }
    }
    else {
        const auto offset = this->getCurrentAccumulatorStep() * stepInputDims.size();
        auto buffer = ArrayTensor<DT, IT>(accumulationBuffer.data() + offset, stepInputDims);
        BL_CHECK(Memory::Copy(buffer, data, stream));
    }

    return Result::SUCCESS;
}

template class BLADE_API Accumulate<Modules::Guppi::Writer<CF16>, Device::CPU, CF16>;
template class BLADE_API Accumulate<Modules::Guppi::Writer<CF32>, Device::CPU, CF32>;

template class BLADE_API Accumulate<Modules::Filterbank::Writer<F16>, Device::CPU, F16>;
template class BLADE_API Accumulate<Modules::Filterbank::Writer<F32>, Device::CPU, F32>;
template class BLADE_API Accumulate<Modules::Filterbank::Writer<F64>, Device::CPU, F64>;

}  // namespace Blade::Pipelines::Generic
