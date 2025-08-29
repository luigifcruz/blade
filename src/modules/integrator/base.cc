#define BL_LOG_DOMAIN "M::INTEGRATOR"

#include <type_traits>
#include <typeindex>

#include "blade/modules/integrator.hh"

#include "integrator.jit.hh"

namespace Blade::Modules {

template<typename IT, typename OT>
Integrator<IT, OT>::Integrator(const Config& config,
                             const Input& input,
                             const Stream& stream)
        : Module(integrator_program),
          config(config),
          input(input),
          computeRatio(config.rate) {
    if constexpr (!std::is_same<IT, OT>::value) {
        BL_FATAL("Input ({}) and output ({}) types aren't the same. Casting isn't supported by Integrator yet.",
                 TypeInfo<IT>::name, TypeInfo<OT>::name);
        BL_CHECK_THROW(Result::ERROR);
    }

    if ((input.buf.shape()[config.axis] % config.size) != 0) {
        BL_FATAL("Input dimension #{} (length of {}) is not divisible by the integration size ({}).",
                 config.axis, input.buf.shape()[config.axis], config.size);
        BL_CHECK_THROW(Result::ERROR);
    }

    if (config.size == 1 && config.rate == 1) {
        BL_INFO("Bypassing integration because size and rate are 1.");
        BL_CHECK_THROW(Link(output.buf, input.buf));
    }
    else {
        U64 integratedElementCount = 1;
        for (int i = config.axis+1; i < 4; i++) {
            integratedElementCount *= input.buf.shape()[i]; 
        }
    
        // Configure kernel instantiation.
        BL_CHECK_THROW(
            this->createKernel(
                // Kernel name.
                "main",
                // Kernel function key.
                "integrator",
                // Kernel grid & block size.
                PadGridSize(
                    getInputBuffer().size() / integratedElementCount / config.size,
                    config.blockSize
                ),
                config.blockSize,
                0,
                // Kernel templates.
                TypeInfo<IT>::name,
                TypeInfo<OT>::name,
                config.size,
                integratedElementCount,
                getInputBuffer().size() / integratedElementCount / config.size
            )
        );
    
        // Allocate output buffers.
        output.buf = ArrayTensor<Device::CUDA, OT>(getOutputBufferShape());
    }


    // Print configuration values.

    BL_INFO("Type: {} -> {}", TypeInfo<IT>::name, TypeInfo<OT>::name);
    BL_INFO("Shape: {} -> {}", getInputBuffer().shape(),
                               getOutputBuffer().shape());
    BL_INFO("Size: {}", config.size);
    BL_INFO("Rate: {}", config.rate);
    BL_INFO("Axis: {}", config.axis);
}

template<typename IT, typename OT>
Result Integrator<IT, OT>::compile(const Stream& stream) {
    if (config.size == 1 && config.rate == 1) {
        return Result::SUCCESS;
    }
    BL_DEBUG("Compiling Integrator Axis {}, Size: {}", config.axis, config.size);
    BL_CHECK(this->compileKernel("main", stream));
    return Result::SUCCESS;
}

template<typename IT, typename OT>
Result Integrator<IT, OT>::process(const U64& currentStepCount, const Stream& stream) {
    if (config.size == 1 && config.rate == 1) {
        return Result::SUCCESS;
    }

    if (currentStepCount == 0) {
        cudaMemsetAsync(output.buf.data(), 0, output.buf.size_bytes(), stream);
    }
    return runKernel("main", stream, input.buf, output.buf);
}

template class BLADE_API Integrator<CI8, CI8>;
template class BLADE_API Integrator<CF16, CF16>;
template class BLADE_API Integrator<CF32, CF32>;
template class BLADE_API Integrator<F32, F32>;

}  // namespace Blade::Modules
