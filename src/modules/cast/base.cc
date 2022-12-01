#define BL_LOG_DOMAIN "M::CAST"

#include <type_traits>
#include <typeindex>

#include "blade/modules/cast.hh"

#include "cast.jit.hh"

namespace Blade::Modules {

template<typename IT, typename OT>
Cast<IT, OT>::Cast(const Config& config, const Input& input)
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
                getInputBuffer().size() * CudaTypeSize<IT>(), 
                config.blockSize
            ),
            config.blockSize,
            // Kernel templates.
            CudaType<IT>(),
            CudaType<OT>(),
            getInputBuffer().size() * CudaTypeSize<IT>()
        )
    );

    // Allocate output buffers.
    BL_CHECK_THROW(output.buf.resize(getOutputBufferDims()));

    // Print configuration values.
    BL_INFO("Type: {} -> {}", TypeInfo<IT>::name, TypeInfo<OT>::name);
    BL_INFO("Dimensions [A, F, T, P]: {} -> {}", getInputBuffer().dims(), 
                                                 getOutputBuffer().dims());
}

template<typename IT, typename OT>
const Result Cast<IT, OT>::process(const cudaStream_t& stream) {
    return this->runKernel("main", stream, input.buf.data(), output.buf.data());
}

template class BLADE_API Cast<CI8, CF32>;
template class BLADE_API Cast<CI8, CF16>;

template class BLADE_API Cast<CF16, F16>;
template class BLADE_API Cast<CF16, F32>;
template class BLADE_API Cast<CF16, CF32>;

template class BLADE_API Cast<CF32, F16>;
template class BLADE_API Cast<CF32, F32>;
template class BLADE_API Cast<CF32, CF16>;

template class BLADE_API Cast<F16, F32>;
template class BLADE_API Cast<F16, CF32>;
template class BLADE_API Cast<F16, CF16>;

template class BLADE_API Cast<F32, F16>;
template class BLADE_API Cast<F32, CF32>;
template class BLADE_API Cast<F32, CF16>;

}  // namespace Blade::Modules
