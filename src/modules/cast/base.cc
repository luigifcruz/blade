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
    BL_INFO("===== Cast Module Configuration");
    BL_INFO("Input Size: {}", config.inputSize);

    auto size = config.inputSize * CudaTypeSize<IT>();
    kernel = Template("cast").instantiate(CudaType<IT>(), CudaType<OT>(), size);
    grid = dim3((size + block.x - 1) / block.x);

    BL_CHECK_THROW(InitInput(input.buf, config.inputSize));
    BL_CHECK_THROW(InitOutput(output.buf, config.inputSize));
}

template<typename IT, typename OT>
Result Cast<IT, OT>::process(const cudaStream_t& stream) {
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

template class BLADE_API Cast<CF16, CF32>;
template class BLADE_API Cast<CF32, CF16>;
template class BLADE_API Cast<CI8, CF32>;

}  // namespace Blade::Modules
