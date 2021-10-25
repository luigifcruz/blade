#include "blade/checker/base.hh"

#include "checker.jit.hh"

namespace Blade::Checker {

Generic::Generic(const Config & config) :
    Kernel(config.blockSize),
    config(config),
    cache(100, *checker_kernel)
{
    BL_DEBUG("Initilizating class.");

    block = dim3(config.blockSize);
}

Generic::~Generic() {
    BL_DEBUG("Destroying class.");
}

template<typename IT, typename OT>
Result Generic::run(IT a, OT b, std::span<std::size_t> &result, std::size_t size,
        std::size_t scale, cudaStream_t cudaStream) {
    auto kernel = Template("checker").instantiate(Type<IT>(), size * scale, scale);
    dim3 grid = dim3(((size * scale) + block.x - 1) / block.x);

    cache
        .get_kernel(kernel)
        ->configure(grid, block, 0, cudaStream)
        ->launch(a, b, result.data());

    BL_CUDA_CHECK_KERNEL([&]{
        BL_FATAL("Kernel failed to execute: {}", err);
        return Result::CUDA_ERROR;
    });

    return Result::SUCCESS;
}


template<typename IT, typename OT>
Result Generic::run(const std::span<std::complex<IT>> &a,
                    const std::span<std::complex<OT>> &b,
                          std::span<std::size_t> &result,
                          cudaStream_t cudaStream) {
    if (a.size() != b.size()) {
        BL_FATAL("Size mismatch between checker inputs.");
        return Result::CUDA_ERROR;
    }

    return this->run(
        reinterpret_cast<const IT*>(a.data()),
        reinterpret_cast<const OT*>(b.data()),
        result, a.size(), 2, cudaStream);
}

template<typename IT, typename OT>
Result Generic::run(const std::span<IT> &a,
                    const std::span<OT> &b,
                          std::span<std::size_t> &result,
                          cudaStream_t cudaStream) {
    if (a.size() != b.size()) {
        BL_FATAL("Size mismatch between checker inputs.");
        return Result::CUDA_ERROR;
    }

    return this->run(a.data(), b.data(), result, a.size(), 1, cudaStream);
}

template Result Generic::run(const std::span<std::complex<float>>&,
                             const std::span<std::complex<float>>&,
                                   std::span<std::size_t> &,
                                   cudaStream_t);

template Result Generic::run(const std::span<std::complex<int8_t>>&,
                             const std::span<std::complex<int8_t>>&,
                                   std::span<std::size_t> &,
                                   cudaStream_t);

template Result Generic::run(const std::span<std::complex<half>>&,
                             const std::span<std::complex<half>>&,
                                   std::span<std::size_t> &,
                                   cudaStream_t);

template Result Generic::run(const std::span<float>&,
                             const std::span<float>&,
                                   std::span<std::size_t> &,
                                   cudaStream_t);

template Result Generic::run(const std::span<int8_t>&,
                             const std::span<int8_t>&,
                                   std::span<std::size_t> &,
                                   cudaStream_t);

template Result Generic::run(const std::span<half>&,
                             const std::span<half>&,
                                   std::span<std::size_t> &,
                                   cudaStream_t);

} // namespace Blade::Checker
