#include <algorithm>
#include <ctime>

#include "blade/checker/base.hh"

using namespace Blade;

Result Init() {
    Checker::Generic checker({8192});

    BL_INFO("Allocating CUDA memory...");
    static std::complex<float>* input_ptr;
    BL_CUDA_CHECK(cudaMallocManaged(&input_ptr, checker.getInputSize() * sizeof(std::complex<float>)), [&]{
        BL_FATAL("Can't allocate complex checker test input buffer: {}", err);
    });

    static std::complex<float>* output_ptr;
    BL_CUDA_CHECK(cudaMallocManaged(&output_ptr, checker.getInputSize() * sizeof(std::complex<float>)), [&]{
        BL_FATAL("Can't allocate complex checker test output buffer: {}", err);
    });

    BL_INFO("Generating test data...");
    std::srand(unsigned(std::time(nullptr)));
    std::span<std::complex<float>> input{input_ptr, checker.getInputSize()};
    std::generate(input.begin(), input.end(), std::rand);

    std::span<std::complex<float>> output{output_ptr, checker.getInputSize()};
    std::generate(output.begin(), output.end(), []{
        return std::complex<float>{1.42, 1.69};
    });

    size_t counter = 0;

    BL_INFO("Running kernels...");
    if ((counter = checker.run(input_ptr, output_ptr)) != checker.getInputSize()) {
        BL_FATAL("[SUBTEST {}] Expected {} matches but found {}.", 0, checker.getInputSize(), counter);
        return Result::ERROR;
    }

    if ((counter = checker.run(input_ptr, input_ptr)) != 0) {
        BL_FATAL("[SUBTEST {}] Expected {} matches but found {}.", 1, 0, counter);
        return Result::ERROR;
    }

    if ((counter = checker.run(output_ptr, output_ptr)) != 0) {
        BL_FATAL("[SUBTEST {}] Expected {} matches but found {}.", 2, 0, counter);
        return Result::ERROR;
    }

    input[0] = output[0];
    input[6] = output[6];

    if ((counter = checker.run(input_ptr, output_ptr)) != checker.getInputSize() - 2) {
        BL_FATAL("[SUBTEST {}] Expected {} matches but found {}.", 3, checker.getInputSize() - 2, counter);
        return Result::ERROR;
    }

    cudaFree(input_ptr);
    cudaFree(output_ptr);

    return Result::SUCCESS;
}

int main() {
    Logger guard{};

    BL_INFO("Welcome to BL Beamformer.");

    if (Init() != Result::SUCCESS) {
        BL_WARN("Fault was encountered. System is exiting...");
        return 1;
    }

    BL_INFO("Nominal shutdown...");

    return 0;
}
