#define BL_LOG_DOMAIN "M::CORRELATOR"

#include "blade/modules/correlator.hh"

#include "correlator.jit.hh"

namespace Blade::Modules {

template<typename IT, typename OT>
Correlator<IT, OT>::Correlator(const Config& config,
                           const Input& input,
                           const Stream& stream)
        : Module(correlator_program),
          config(config),
          input(input),
          computeRatio(config.integrationRate) {

    // Check configuration values.

    if (getInputBuffer().shape().numberOfAspects() <= 0) {
        BL_FATAL("Number of aspects ({}) should be more than zero.",
                 getInputBuffer().shape().numberOfAspects());
        BL_CHECK_THROW(Result::ERROR);
    }

    if (getInputBuffer().shape().numberOfFrequencyChannels() <= 0) {
        BL_FATAL("Number of frequency channels ({}) should be more than zero.",
                 getInputBuffer().shape().numberOfFrequencyChannels());
        BL_CHECK_THROW(Result::ERROR);
    }

    if (config.integrationRate < 1) {
        BL_FATAL("Integration size ({}) should be one (1) or more.", config.integrationRate);
        BL_CHECK_THROW(Result::ERROR);
    }

    if (config.conjugateAntennaIndex > 1) {
        BL_FATAL("Conjugate antenna index ({}) should be zero (0) for Antenna A or one (1) for Antenna B.",
                 config.conjugateAntennaIndex);
        BL_CHECK_THROW(Result::ERROR);
    }

    if (getInputBuffer().shape().numberOfPolarizations() != 2) {
        BL_FATAL("Number of polarizations ({}) should be two. Feature not implemented.",
                 getInputBuffer().shape().numberOfPolarizations());
        BL_CHECK_THROW(Result::ERROR);
    }

    // Block size settings calculation.

    const bool optimizeTimeDomain = (getInputBuffer().shape().numberOfTimeSamples() >
                                     getInputBuffer().shape().numberOfFrequencyChannels());

    const U64 BLOCK_X = (optimizeTimeDomain) ? 1 : config.blockSize;
    const U64 BLOCK_Y = (optimizeTimeDomain) ? config.blockSize : 1;

    // Process calculation mode.

    const std::string calculationDataType = [&]{
        switch (config.calculationMode) {
            case CALC_MODE::INTEGER:
                return TypeInfo<CI32>::name;
            case CALC_MODE::SINGLE_PRECISION_FP:
                return TypeInfo<CF32>::name;
            case CALC_MODE::DOUBLE_PRECISION_FP:
                return TypeInfo<CF64>::name;
            default:
                BL_FATAL("Unsupported calculation mode.");
                BL_CHECK_THROW(Result::ERROR);
        }

        return "";
    }();

    // Check block size configuration values.

    if ((getInputBuffer().shape().numberOfFrequencyChannels() % BLOCK_X) != 0) {
        BL_FATAL("Number of frequency channels ({}) should be divisible by block size ({}).",
                 getInputBuffer().shape().numberOfFrequencyChannels(), BLOCK_X);
        BL_CHECK_THROW(Result::ERROR);
    }

    if ((getInputBuffer().shape().numberOfTimeSamples() % BLOCK_Y) != 0) {
        BL_FATAL("Number of time sample ({}) should be divisible by block size ({}).",
                 getInputBuffer().shape().numberOfTimeSamples(), BLOCK_Y);
        BL_CHECK_THROW(Result::ERROR);
    }

    // Configure kernel instantiation.

    BL_CHECK_THROW(
        createKernel(
            // Kernel name.
            "main",
            // Kernel function key.
            "correlator",
            // Kernel grid & block size.
            dim3(
                getInputBuffer().shape().numberOfAspects(),
                getInputBuffer().shape().numberOfFrequencyChannels() / BLOCK_X
            ),
            dim3(
                BLOCK_X,
                BLOCK_Y
            ),
            getInputBuffer().shape().numberOfTimeSamples() *
            getInputBuffer().shape().numberOfPolarizations() *
            sizeof(IT) *
            ((config.useSharedMemory) ? 1 : 0),
            // Kernel templates.
            TypeInfo<IT>::name,
            TypeInfo<OT>::name,
            calculationDataType,
            getInputBuffer().shape().numberOfAspects(),
            getInputBuffer().shape().numberOfFrequencyChannels(),
            getInputBuffer().shape().numberOfTimeSamples(),
            getInputBuffer().shape().numberOfPolarizations(),
            BLOCK_X,
            BLOCK_Y,
            config.conjugateAntennaIndex,
            config.useSharedMemory
        )
    );

    // Allocate output bffers.
    output.buf = ArrayTensor<Device::CUDA, OT>(getOutputBufferShape());

    // Print configuration values.
    BL_INFO("Type: {} -> {}", TypeInfo<IT>::name, TypeInfo<OT>::name);
    BL_INFO("Shape: {} -> {}", getInputBuffer().shape(),
                               getOutputBuffer().shape());
    BL_INFO("Integration Rate: {}", config.integrationRate);
    BL_INFO("Memory Mode: {}", (config.useSharedMemory) ? "Shared Memory" : "Global Memory");
    BL_INFO("Optimized Domain: {}", (optimizeTimeDomain) ? "Time Domain" : "Frequency Domain");
    BL_INFO("Calculation Mode: {}", config.calculationMode);
    BL_INFO("Antenna Conjugation: {}", (config.conjugateAntennaIndex) ? "Antenna B" : "Antenna A");
}

template<typename IT, typename OT>
Result Correlator<IT, OT>::process(const U64& currentStepCount, const Stream& stream) {
    if (currentStepCount == 0) {
        cudaMemsetAsync(output.buf.data(), 0, output.buf.size_bytes(), stream);
    }
    return runKernel("main", stream, input.buf, output.buf);
}

template class BLADE_API Correlator<CI8, CF32>;
template class BLADE_API Correlator<CF32, CF32>;

}  // namespace Blade::Modules
