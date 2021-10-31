#include "blade/checker/base.hh"

#include "checker.jit.hh"

namespace Blade {

Checker::Checker(const Config& config) :
    Kernel(config.blockSize), config(config), cache(100, *checker_kernel) {
    BL_DEBUG("Initilizating class.");

    block = dim3(config.blockSize);

    if (cudaMallocManaged(&counter, sizeof(unsigned long long int)) != cudaSuccess) {
        BL_FATAL("Can't allocate CUDA memory for counter.");
        throw Result::ERROR;
    }
    *counter = 0;
}

Checker::~Checker() {
    BL_DEBUG("Destroying class.");
    cudaFree(counter);
}

template<typename IT, typename OT>
unsigned long long int Checker::run(IT a, OT b, std::size_t size, std::size_t scale,
        cudaStream_t cudaStream) {
    auto kernel = Template("checker").instantiate(Type<IT>(), size * scale);
    dim3 grid = dim3(((size * scale) + block.x - 1) / block.x);

    *counter = 0;
    cache
        .get_kernel(kernel)
        ->configure(grid, block, 0, cudaStream)
        ->launch(a, b, counter);

    BL_CUDA_CHECK_KERNEL([&]{
        BL_FATAL("Kernel failed to execute: {}", err);
        return -1;
    });

    cudaDeviceSynchronize();

    return (*counter) / scale;
}


template<typename IT, typename OT>
unsigned long long int Checker::run(const std::span<std::complex<IT>>& a,
                                    const std::span<std::complex<OT>>& b,
                                          cudaStream_t cudaStream) {
    if (a.size() != b.size()) {
        BL_FATAL("Size mismatch between checker inputs.");
        return -1;
    }

    return this->run(
        reinterpret_cast<const IT*>(a.data()),
        reinterpret_cast<const OT*>(b.data()),
        a.size(), 2);
}

template<typename IT, typename OT>
unsigned long long int Checker::run(const std::span<IT>& a,
                                    const std::span<OT>& b,
                                          cudaStream_t cudaStream) {
    if (a.size() != b.size()) {
        BL_FATAL("Size mismatch between checker inputs.");
        return -1;
    }

    return this->run(a.data(), b.data(), a.size());
}

template unsigned long long int Checker::run(const std::span<CF32>&,
                                             const std::span<CF32>&,
                                                   cudaStream_t);

template unsigned long long int Checker::run(const std::span<CI8>&,
                                             const std::span<CI8>&,
                                                   cudaStream_t);

template unsigned long long int Checker::run(const std::span<CF16>&,
                                             const std::span<CF16>&,
                                                   cudaStream_t);

template unsigned long long int Checker::run(const std::span<F32>&,
                                             const std::span<F32>&,
                                                   cudaStream_t);

template unsigned long long int Checker::run(const std::span<I8>&,
                                             const std::span<I8>&,
                                                   cudaStream_t);

template unsigned long long int Checker::run(const std::span<F16>&,
                                             const std::span<F16>&,
                                                   cudaStream_t);

}  // namespace Blade
