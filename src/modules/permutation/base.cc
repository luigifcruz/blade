#define BL_LOG_DOMAIN "M::PERMUTATION"

#include <type_traits>
#include <typeindex>

#include "blade/modules/permutation.hh"

#include "permutation.jit.hh"

namespace Blade::Modules {

template<typename IT, typename OT>
Permutation<IT, OT>::Permutation(const Config& config,
                       const Input& input,
                       const Stream& stream)
        : Module(permutation_program),
          config(config),
          input(input) {
    if constexpr (std::is_same<IT, OT>::value) {
        BL_FATAL("Input ({}) and output ({}) types aren't the same. Casting isn't supported by Permutation yet.",
                 TypeInfo<IT>::name, TypeInfo<OT>::name);
        BL_CHECK_THROW(Result::ERROR);
    }

    // Configure kernels.
    BL_CHECK_THROW(
        this->createKernel(
            // Kernel name.
            "main",
            // Kernel function key.
            "permutation",
            // Kernel grid & block size.
            PadGridSize(
                getInputBuffer().size() * TypeInfo<IT>::cudaSize,
                config.blockSize
            ),
            config.blockSize,
            // Kernel templates.
            TypeInfo<typename TypeInfo<IT>::subtype>::cudaName
        )
    );

    // Allocate output buffers.
    output.buf = ArrayTensor<Device::CUDA, OT>(getOutputBufferShape());

    // Print configuration values.
    BL_INFO("Type: {} -> {}", TypeInfo<IT>::name, TypeInfo<OT>::name);
    BL_INFO("Shape: {} -> {}", getInputBuffer().shape(), 
                               getOutputBuffer().shape());
    BL_INFO("Indexes: {}", config.indexes);
}

template<typename IT, typename OT>
Result Permutation<IT, OT>::process(const U64& currentStepCount, const Stream& stream) {
    return this->runKernel("main", stream, input.buf.data(), output.buf.data(), getInputBuffer().shape());
}

template class BLADE_API Permutation<CI8, CI8>;
template class BLADE_API Permutation<CF16, CF16>;
template class BLADE_API Permutation<CF32, CF32>;

}  // namespace Blade::Modules
