#include <type_traits>

#include "blade/cast/base.hh"

#include "cast.jit.hh"

namespace Blade::Cast {

Generic::Generic(const Config & config) :
    Kernel(config.blockSize),
    config(config),
    cache(100, *cast_kernel)
{
    BL_DEBUG("Initilizating class.");

    block = dim3(config.blockSize);
}

Generic::~Generic() {
    BL_DEBUG("Destroying class.");
}

template<typename OT, typename IT>
Result Generic::run(IT input, OT output, std::size_t size, cudaStream_t cudaStream) {
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

    return Result::SUCCESS;
}

Result Generic::run(const std::span<std::complex<int8_t>> &input,
                          std::span<std::complex<float>> &output,
                          cudaStream_t cudaStream) {
    if (input.size() != output.size()) {
        BL_FATAL("Size mismatch between input and output ({}, {}).",
                input.size(), output.size());
        return Result::ASSERTION_ERROR;
    }

    return this->run(
        reinterpret_cast<const int8_t*>(input.data()),
        reinterpret_cast<float*>(output.data()),
        input.size() * 2,
        cudaStream
    );
}

} // namespace Blade::Cast
