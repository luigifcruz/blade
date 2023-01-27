#define BL_LOG_DOMAIN "M::DETECTOR"

#include "blade/modules/detector.hh"

#include "detector.jit.hh"

namespace Blade::Modules {

template<typename IT, typename OT>
Detector<IT, OT>::Detector(const Config& config, const Input& input)
        : Module(detector_program),
          config(config),
          input(input),
          apparentIntegrationSize(config.integrationSize) {
    // Check configuration values.
    if (apparentIntegrationSize <= 0) {
        BL_WARN("Integration size ({}) should be more than zero.", apparentIntegrationSize);
        BL_CHECK_THROW(Result::ERROR);
    }

    if (getInputBuffer().dims().numberOfTimeSamples() > 1) {
        if ((getInputBuffer().dims().numberOfTimeSamples() % apparentIntegrationSize) != 0) {
            BL_FATAL("The number of time samples ({}) should be divisable "
                     "by the integration size ({}).",
                     getInputBuffer().dims().numberOfTimeSamples(),
                     apparentIntegrationSize);
            BL_CHECK_THROW(Result::ERROR);
        }
    }

    if (getInputBuffer().dims().numberOfPolarizations() != 2) {
        BL_FATAL("Number of polarizations ({}) should be two (2).", 
                 getInputBuffer().dims().numberOfPolarizations());
        BL_CHECK_THROW(Result::ERROR);
    }

    if (getInputBuffer().dims().numberOfAspects() <= 0) {
        BL_FATAL("Number of aspects ({}) should be more than zero.", 
                 getInputBuffer().dims().numberOfAspects());
        BL_CHECK_THROW(Result::ERROR);
    }

    // Configure kernel instantiation.
    std::string kernel_key;
    switch (config.kernel) {
        case DetectorKernel::AFTP_4pol:
            kernel_key = "detector_4pol_AFTP";
            break;
        case DetectorKernel::AFTP_1pol:
            kernel_key = "detector_1pol_AFTP";
            break;
        case DetectorKernel::ATPF_4pol:
            kernel_key = "detector_4pol_ATPF";
            break;
        case DetectorKernel::ATPF_1pol:
            kernel_key = "detector_1pol_ATPF";
            break;
        case DetectorKernel::ATPFrev_4pol:
            kernel_key = "detector_4pol_ATPFrev";
            break;
        case DetectorKernel::ATPFrev_1pol:
            kernel_key = "detector_1pol_ATPFrev";
            break;
        default:
            BL_FATAL("Detector kernel enumeration ({}) not supported.", 
                (int) config.kernel);
            BL_CHECK_THROW(Result::ERROR);
    }

    if (getInputBuffer().dims().numberOfTimeSamples() < config.integrationSize) {
        apparentIntegrationSize = 1;
        BL_INFO("Integration Procedure: Stepped");
    } else {
        BL_INFO("Integration Procedure: Blockwise");
    }

    BL_CHECK_THROW(
        createKernel(
            // Kernel name.
            "main",
            // Kernel function key.
            kernel_key,
            // Kernel grid & block size.
            PadGridSize(
                getInputBuffer().size() / 
                    getInputBuffer().dims().numberOfPolarizations(),
                config.blockSize
            ),
            config.blockSize,
            // Kernel templates.
            getInputBuffer().dims().numberOfAspects(),
            getInputBuffer().dims().numberOfFrequencyChannels(),
            getInputBuffer().dims().numberOfTimeSamples(),
            apparentIntegrationSize
        )
    );

    // Allocate output buffers.
    BL_CHECK_THROW(output.buf.resize(getOutputBufferDims()));
    BL_CHECK_THROW(ctrlResetTensor.resize({1}));

    // Set default values.
    ctrlResetTensor[0] = true;

    // Print configuration values.
    BL_INFO("Type: {} -> {}", TypeInfo<IT>::name, TypeInfo<OT>::name);
    BL_INFO("Dimensions [A, F, T, P]: {} -> {}", getInputBuffer().dims(), getOutputBuffer().dims());
    BL_INFO("Integration Size: {}", config.integrationSize);
    BL_INFO("Kernel operation: {}", DetectorKernelName(config.kernel));
}

template<typename IT, typename OT>
const Result Detector<IT, OT>::preprocess(const cudaStream_t& stream,
                                          const U64& currentComputeCount) {
    if (config.integrationSize == apparentIntegrationSize) {
        return Result::SUCCESS;
    }

    if ((currentComputeCount % config.integrationSize) == 0) {
        ctrlResetTensor[0] = true;
    } else {
        ctrlResetTensor[0] = false;
    }

    return Result::SUCCESS;
}

template<typename IT, typename OT>
const Result Detector<IT, OT>::process(const cudaStream_t& stream) {
    return runKernel(
        // Kernel name.
        "main",
        // Kernel stream.
        stream, 
        // Kernel arguments.
        input.buf.data(),
        output.buf.data(),
        ctrlResetTensor.data()
    );
}

template class BLADE_API Detector<CF32, F32>;

}  // namespace Blade::Modules
