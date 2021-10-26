#include <type_traits>

#include "blade/cast/base.hh"

#include "cast.jit.hh"

namespace Blade {

Cast::Cast(const Config& config) :
    Kernel(config.blockSize),
    config(config),
    cache(100, *cast_kernel)
{
    BL_DEBUG("Initilizating class.");

    block = dim3(config.blockSize);
}

Cast::~Cast() {
    BL_DEBUG("Destroying class.");
}

template<typename OT, typename IT>
Result Cast::run(IT input, OT output, std::size_t size, cudaStream_t cudaStream) {
    dim3 grid = dim3((size + block.x - 1) / block.x);
    auto kernel = Template("cast").instantiate(
        Type<typename std::remove_pointer<IT>::type>(),
        Type<typename std::remove_pointer<OT>::type>(),
        size
    );

    cache
        .get_kernel(kernel)
        ->configure(grid, block, 0, cudaStream)
        ->launch(input, output);

    BL_CUDA_CHECK_KERNEL([&]{
        BL_FATAL("Kernel failed to execute: {}", err);
        return Result::CUDA_ERROR;
    });

    return Result::SUCCESS;
}

template<typename IT, typename OT>
Result Cast::run(const std::span<std::complex<IT>>& input,
                          std::span<std::complex<OT>>& output,
                          cudaStream_t cudaStream) {
    if (input.size() != output.size()) {
        BL_FATAL("Size mismatch between input and output ({}, {}).",
                input.size(), output.size());
        return Result::ASSERTION_ERROR;
    }

    return this->run(
        reinterpret_cast<const IT*>(input.data()),
        reinterpret_cast<OT*>(output.data()),
        input.size() * 2,
        cudaStream
    );
}

template Result Cast::run(const std::span<std::complex<int8_t>>&,
                                   std::span<std::complex<float>>&,
                                   cudaStream_t);

template Result Cast::run(const std::span<std::complex<float>>&,
                                   std::span<std::complex<half>>&,
                                   cudaStream_t);

} // namespace Blade
