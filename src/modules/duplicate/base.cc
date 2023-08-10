#define BL_LOG_DOMAIN "M::DUPLICATE"

#include <type_traits>
#include <typeindex>

#include "blade/modules/duplicate.hh"

#include "duplicate.jit.hh"

namespace Blade::Modules {

template<typename IT, typename OT>
Duplicate<IT, OT>::Duplicate(const Config& config,
                             const Input& input,
                             const cudaStream_t& stream)
        : Module(duplicate_program),
          config(config),
          input(input) {
    if constexpr (std::is_same<IT, OT>::value) {
        BL_FATAL("Input ({}) and output ({}) types aren't the same. Casting isn't supported by Duplicate yet.",
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
Result Duplicate<IT, OT>::process(const U64& currentStepCount, const cudaStream_t& stream) {
    return Blade::Copy(output.buf, input.buf, stream);
}

template class BLADE_API Duplicate<CI8, CI8>;
template class BLADE_API Duplicate<CF16, CF16>;
template class BLADE_API Duplicate<CF32, CF32>;

}  // namespace Blade::Modules
