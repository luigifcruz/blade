#include "blade/utils/checker.hh"

using namespace Blade;

Result Init(std::size_t testSize = 8192) {
    BL_INFO("Allocating CUDA memory...");
    static CF32* input_ptr;
    static CF32* output_ptr;

    BL_CUDA_CHECK(cudaMallocManaged(&input_ptr, testSize *
                sizeof(CF32)), [&]{
        BL_FATAL("Can't allocate complex checker test input buffer: {}", err);
    });

    BL_CUDA_CHECK(cudaMallocManaged(&output_ptr, testSize *
                sizeof(CF32)), [&]{
        BL_FATAL("Can't allocate complex checker test output buffer: {}", err);
    });

    std::span input{input_ptr, testSize};
    std::span output{output_ptr, testSize};

    BL_INFO("Generating test data...");
    std::srand(unsigned(std::time(nullptr)));

    for (auto& element : input) {
        element = {
            static_cast<F32>(std::rand()),
            static_cast<F32>(std::rand())
        };
    }

    for (auto& element : output) {
        element = std::complex(1.42, 1.69);
    }

    BL_INFO("Running kernels...");
    size_t counter = 0;

    if ((counter = Checker::run(input, output)) != testSize) {
        BL_FATAL("[SUBTEST {}] Expected {} matches but found {}.",
                0, testSize, counter);
        return Result::ERROR;
    }

    if ((counter = Checker::run(input, input)) != 0) {
        BL_FATAL("[SUBTEST {}] Expected {} matches but found {}.",
                1, 0, counter);
        return Result::ERROR;
    }

    if ((counter = Checker::run(output, output)) != 0) {
        BL_FATAL("[SUBTEST {}] Expected {} matches but found {}.",
                2, 0, counter);
        return Result::ERROR;
    }

    input[0] = output[0];
    input[6] = output[6];

    if ((counter = Checker::run(input, output)) != testSize - 2) {
        BL_FATAL("[SUBTEST {}] Expected {} matches but found {}.",
                3, testSize - 2, counter);
        return Result::ERROR;
    }

    cudaFree(input_ptr);
    cudaFree(output_ptr);

    return Result::SUCCESS;
}

int main() {
    Logger guard{};

    BL_INFO("Testing complex checker.");

    if (Init() != Result::SUCCESS) {
        BL_WARN("Fault was encountered. Test is exiting...");
        return 1;
    }

    BL_INFO("Test succeeded.");

    return 0;
}
