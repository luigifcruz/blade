#include <span>
#include <algorithm>
#include <ctime>

#include "bl-beamformer/base.hh"

using namespace BL;

Result Init() {
    Checker checker({8192});

    static std::complex<float>* input_ptr;
    BL_CUDA_CHECK(cudaMallocManaged(&input_ptr, checker.inputLen() * sizeof(std::complex<float>)), [&]{
        BL_FATAL("Can't allocate complex checker test input buffer.");
    });

    static std::complex<float>* output_ptr;
    BL_CUDA_CHECK(cudaMallocManaged(&output_ptr, checker.inputLen() * sizeof(std::complex<float>)), [&]{
        BL_FATAL("Can't allocate complex checker test output buffer.");
    });

    std::srand(unsigned(std::time(nullptr)));
    std::span<std::complex<float>> input{input_ptr, checker.inputLen()};
    std::generate(input.begin(), input.end(), std::rand);

    std::span<std::complex<float>> output{output_ptr, checker.inputLen()};
    std::generate(output.begin(), output.end(), []{
        return std::complex<float>{1.42, 1.69};
    });

    size_t counter = 0;

    if ((counter = checker.run(input_ptr, output_ptr)) != checker.inputLen()) {
        BL_FATAL("[SUBTEST {}] Expected {} matches but found {}.", 0, checker.inputLen(), counter);
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

    if ((counter = checker.run(input_ptr, output_ptr)) != checker.inputLen() - 2) {
        BL_FATAL("[SUBTEST {}] Expected {} matches but found {}.", 3, checker.inputLen() - 2, counter);
        return Result::ERROR;
    }

    cudaFree(input_ptr);
    cudaFree(output_ptr);

    return Result::SUCCESS;
}

int main() {
    Logger::Init();

    BL_INFO("Welcome to BL Beamformer.");

    if (Init() != Result::SUCCESS) {
        BL_WARN("Fault was encountered. System is exiting...");
        return 1;
    }

    BL_INFO("Nominal shutdown...");
    Logger::Shutdown();

    return 0;
}
