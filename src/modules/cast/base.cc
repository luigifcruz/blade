#define BL_LOG_DOMAIN "M::CAST"

#include <type_traits>
#include <typeindex>

#include "blade/modules/cast.hh"

#include "cast.jit.hh"

namespace Blade::Modules {

template<typename IT, typename OT>
Cast<IT, OT>::Cast(const Config& config,
                   const Input& input,
                   const Stream& stream)
        : Module(cast_program),
          config(config),
          input(input) {
    // Configure kernel instantiation.
    BL_CHECK_THROW(
        this->createKernel(
            // Kernel name.
            "main",
            // Kernel function key.
            "cast",
            // Kernel grid & block size.
            PadGridSize(
                getInputBuffer().size() * TypeInfo<IT>::cudaSize,
                config.blockSize
            ),
            config.blockSize,
            // Kernel templates.
            TypeInfo<typename TypeInfo<IT>::subtype>::cudaName,
            TypeInfo<typename TypeInfo<OT>::subtype>::cudaName,
            getInputBuffer().size() * TypeInfo<IT>::cudaSize
        )
    );

    if ((TypeInfo<IT>::is_complex and !TypeInfo<OT>::is_complex) or 
        (!TypeInfo<IT>::is_complex and TypeInfo<OT>::is_complex)) {
        BL_FATAL("Cannot cast between complex and non-complex types.");
        BL_CHECK_THROW(Result::ERROR);
    }

    if constexpr (std::is_same<IT, OT>::value) {
        BL_DEBUG("Bypassing cast because input and output types are the same.");
        BL_CHECK_THROW(Link(output.buf, input.buf));
        return;
    }

    // Allocate output buffers.
    output.buf = ArrayTensor<Device::CUDA, OT>(getOutputBufferShape());

    // Print configuration values.
    BL_INFO("Type: {} -> {}", TypeInfo<IT>::name, TypeInfo<OT>::name);
    BL_INFO("Shape: {} -> {}", getInputBuffer().shape(), 
                               getOutputBuffer().shape());
}

template<typename IT, typename OT>
Result Cast<IT, OT>::process(const U64& currentStepCount, const Stream& stream) {
    if constexpr (std::is_same<IT, OT>::value) {
        return Result::SUCCESS;
    }
    return this->runKernel("main", stream, input.buf.data(), output.buf.data());
}

// I8 -> X
template class BLADE_API Cast<I8, I8>;
template class BLADE_API Cast<I8, F32>;

// F16 -> X
template class BLADE_API Cast<F16, F16>;
template class BLADE_API Cast<F16, F32>;
template class BLADE_API Cast<F16, CF32>;
template class BLADE_API Cast<F16, CF16>;

// F32 -> X
template class BLADE_API Cast<F32, F32>;
template class BLADE_API Cast<F32, F16>;
template class BLADE_API Cast<F32, CF32>;
template class BLADE_API Cast<F32, CF16>;

// CI8 -> X
template class BLADE_API Cast<CI8, CI8>;
template class BLADE_API Cast<CI8, CF32>;
template class BLADE_API Cast<CI8, CF16>;

// CF32 -> X
template class BLADE_API Cast<CF32, F16>;
template class BLADE_API Cast<CF32, F32>;
template class BLADE_API Cast<CF32, CF16>;
template class BLADE_API Cast<CF32, CF32>;

// CF16 -> X
template class BLADE_API Cast<CF16, F16>;
template class BLADE_API Cast<CF16, F32>;
template class BLADE_API Cast<CF16, CF32>;
template class BLADE_API Cast<CF16, CF16>;

}  // namespace Blade::Modules
