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

    if (cudaMallocManaged(&counter, sizeof(unsigned long long int)) != cudaSuccess) {
        BL_FATAL("Can't allocate CUDA memory for counter.");
        throw Result::ERROR;
    }
    *counter = 0;
}

Generic::~Generic() {
    BL_DEBUG("Destroying class.");
    cudaFree(counter);
}

template<typename IT, typename OT>
unsigned long long int Generic::generic_run(IT a, OT b,
                                            std::size_t size, std::size_t scale) {
    auto kernel = Template("checker").instantiate(Type<IT>(), size * scale);
    dim3 grid = dim3(((size * scale) + block.x - 1) / block.x);

    *counter = 0;
    cache
        .get_kernel(kernel)
        ->configure(grid, block)
        ->launch(a, b, counter);

    BL_CUDA_CHECK_KERNEL([&]{
        BL_FATAL("Kernel failed to execute: {}", err);
        return -1;
    });

    cudaDeviceSynchronize();

    return (*counter) / scale;
}


unsigned long long int Generic::run(const std::span<std::complex<float>> &a,
                                    const std::span<std::complex<float>> &b) {
    if (a.size() != b.size()) {
        BL_FATAL("Size mismatch between checker inputs.");
        return -1;
    }

    return this->generic_run(
        reinterpret_cast<const float*>(a.data()),
        reinterpret_cast<const float*>(b.data()),
        a.size(),
        2
    );
}

unsigned long long int Generic::run(const std::span<std::complex<int8_t>> &a,
                                    const std::span<std::complex<int8_t>> &b) {
    if (a.size() != b.size()) {
        BL_FATAL("Size mismatch between checker inputs.");
        return -1;
    }

    return this->generic_run(
        reinterpret_cast<const int8_t*>(a.data()),
        reinterpret_cast<const int8_t*>(b.data()),
        a.size(),
        2
    );
}

unsigned long long int Generic::run(const std::span<std::complex<half>> &a,
                                    const std::span<std::complex<half>> &b) {
    if (a.size() != b.size()) {
        BL_FATAL("Size mismatch between checker inputs.");
        return -1;
    }

    return this->generic_run(
        reinterpret_cast<const half*>(a.data()),
        reinterpret_cast<const half*>(b.data()),
        a.size(),
        2
    );
}

unsigned long long int Generic::run(const std::span<float> &a,
                                    const std::span<float> &b) {
    if (a.size() != b.size()) {
        BL_FATAL("Size mismatch between checker inputs.");
        return -1;
    }

    return this->generic_run(a.data(), b.data(), a.size());
}

unsigned long long int Generic::run(const std::span<int8_t> &a,
                                    const std::span<int8_t> &b) {
    if (a.size() != b.size()) {
        BL_FATAL("Size mismatch between checker inputs.");
        return -1;
    }

    return this->generic_run(a.data(), b.data(), a.size());
}

} // namespace Blade::Checker
