#define BL_LOG_DOMAIN "M::COPY"

#include <type_traits>
#include <typeindex>

#include "blade/modules/copy.hh"

#include "copy.jit.hh"

namespace Blade::Modules {

template<typename IT, typename OT>
Copy<IT, OT>::Copy(const Config& config,
                       const Input& input,
                       const cudaStream_t& stream)
        : Module(copy_program),
          config(config),
          input(input) {
    if constexpr (std::is_same<IT, OT>::value) {
        BL_FATAL("Input ({}) and output ({}) types aren't the same. Casting isn't supported by Copy yet.",
                 TypeInfo<IT>::name, TypeInfo<OT>::name);
        BL_CHECK_THROW(Result::ERROR);
    }

    // Allocate output buffers.
    output.buf = ArrayTensor<Device::CUDA, OT>(input.buf.shape());

    // Print configuration values.
    BL_INFO("Type: {} -> {}", TypeInfo<IT>::name, TypeInfo<OT>::name);
    BL_INFO("Shape: {} -> {}", getInputBuffer().shape(), 
                               getOutputBuffer().shape());
}

template<typename IT, typename OT>
Result Copy<IT, OT>::process(const cudaStream_t& stream, const U64& currentStepCount) {
    return Blade::Copy(output.buf, input.buf, stream);
}

template class BLADE_API Copy<CI8, CI8>;
template class BLADE_API Copy<CF16, CF16>;
template class BLADE_API Copy<CF32, CF32>;

}  // namespace Blade::Modules
