#include <algorithm>
#include <ctime>

#include "blade/checker/base.hh"

using namespace Blade;

Result Init() {
    Checker checker({8192});

    static int8_t* input_ptr;
    BL_CUDA_CHECK(cudaMallocManaged(&input_ptr, checker.inputLen() * sizeof(int8_t)), [&]{
        BL_FATAL("Can't allocate complex checker test input buffer: {}", err);
    });

    static int8_t* output_ptr;
    BL_CUDA_CHECK(cudaMallocManaged(&output_ptr, checker.inputLen() * sizeof(int8_t)), [&]{
        BL_FATAL("Can't allocate complex checker test output buffer: {}", err);
    });

    std::srand(unsigned(std::time(nullptr)));
    std::span<int8_t> input{input_ptr, checker.inputLen()};
    std::generate(input.begin(), input.end(), std::rand);

    std::span<int8_t> output{output_ptr, checker.inputLen()};
    std::generate(output.begin(), output.end(), []{
        return 69;
    });

    size_t counter = 0;

    if ((counter = checker.run(input_ptr, output_ptr)) < checker.inputLen() - 100) {
        BL_FATAL("[SUBTEST {}] Expected over than {} matches but found {}.", 0, checker.inputLen() - 100, counter);
        return Result::ERROR;
    }

    if ((counter = checker.run(input_ptr, input_ptr)) > 100) {
        BL_FATAL("[SUBTEST {}] Expected less than {} matches but found {}.", 1, 100, counter);
        return Result::ERROR;
    }

    if ((counter = checker.run(output_ptr, output_ptr)) > 100) {
        BL_FATAL("[SUBTEST {}] Expected less than {} matches but found {}.", 2, 100, counter);
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
