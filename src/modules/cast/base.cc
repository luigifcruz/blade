#define BL_LOG_DOMAIN "M::CAST"

#include <type_traits>
#include <typeindex>

#include "blade/modules/cast.hh"

#include "cast.jit.hh"

namespace Blade::Modules {

template<typename IT, typename OT>
Cast<IT, OT>::Cast(const Config& config, const Input& input)
        : Module(config.blockSize, cast_kernel),
          config(config),
          input(input) {
    // Check configuration values.
    auto size = getInputBuffer().size() * CudaTypeSize<IT>();
    kernel = Template("cast").instantiate(CudaType<IT>(), CudaType<OT>(), size);
    grid = dim3((size + block.x - 1) / block.x);

    // Allocate output buffers.
    BL_CHECK_THROW(output.buf.resize(getOutputBufferDims()));

    // Print configuration values.
    BL_INFO("Type: {} -> {}", TypeInfo<IT>::name, TypeInfo<OT>::name);
    BL_INFO("Dimensions [A, F, T, P]: {} -> {}", getInputBuffer().dims(), 
                                                 getOutputBuffer().dims());
}

template<typename IT, typename OT>
const Result Cast<IT, OT>::process(const cudaStream_t& stream) {
    cache
        .get_kernel(kernel)
        ->configure(grid, block, 0, stream)
        ->launch(input.buf.data(), output.buf.data());

    BL_CUDA_CHECK_KERNEL([&]{
        BL_FATAL("Module failed to execute: {}", err);
        return Result::CUDA_ERROR;
    });

    return Result::SUCCESS;
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
