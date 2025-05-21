#define BL_LOG_DOMAIN "M::KURTOSIS"

#include <type_traits>
#include <typeindex>

#include "blade/modules/kurtosis.hh"

#include "kurtosis.jit.hh"

namespace Blade::Modules {

template<typename IT, typename OT>
Kurtosis<IT, OT>::Kurtosis(const Config& config, 
                             const Input& input, 
                             const Stream& stream)
        : Module(kurtosis_program),
          config(config),
          input(input) {
    // Configure kernel instantiation.
    BL_CHECK_THROW(
        this->createKernel(
            // Kernel name.
            std::string("main"),
            // Kernel function key.
            std::string("compute_sk_array"),
            // Kernel grid & block size.
            /*
            PadGridSize(
                getInputBuffer().size(),
                config.blockSize
            ),
            */
            dim3( // grid dimensions (?)
                config.blockSize
            ),
            dim3( // threads per block
                28,
                2
            ),
            0,
            // Kernel templates.
            TypeInfo<IT>::name,
            TypeInfo<OT>::name,
            config.debugMode
        )
    );

    if constexpr (!std::is_same<IT, OT>::value) {
        BL_FATAL("This module requires the type of the input "
                 "({}) and output ({}) to be the same.",
                 TypeInfo<IT>::name, TypeInfo<OT>::name); 
        BL_INFO("Contact the maintainer if this "
                "functionality is required.");
        BL_CHECK_THROW(Result::ERROR);
    }

    // Link output buffers.
    // if (config.inputPolarization == config.outputPolarization) {
    //     BL_INFO("Bypass: Enabled");
    // }

    // Link output buffer or link input with output.
    BL_CHECK_THROW(Link(output.buf, input.buf));

    // Print configuration values.
    BL_INFO("Type: {} -> {}", TypeInfo<IT>::name, TypeInfo<OT>::name);
    BL_INFO("Shape: {} -> {}", getInputBuffer().shape(), 
                              getOutputBuffer().shape());
}

template<typename IT, typename OT>
Result Kurtosis<IT, OT>::process(const U64& currentStepCount, const Stream& stream) {
    /*
    if (config.inputPolarization == config.outputPolarization) {
        return Result::SUCCESS;
    }
    */

    return this->runKernel("main", 
            stream, 
            input.buf.data(), 
            getInputBuffer().shape().numberOfAspects(), 
            getInputBuffer().shape().numberOfFrequencyChannels(), 
            getInputBuffer().shape().numberOfTimeSamples(), 
            getInputBuffer().shape().numberOfPolarizations()
        );
}

template class BLADE_API Kurtosis<CF32, CF32>;
template class BLADE_API Kurtosis<CF16, CF16>;

}  // namespace Blade::Modules
