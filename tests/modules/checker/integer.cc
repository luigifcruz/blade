#include "blade/utils/checker.hh"

using namespace Blade;

Result Init(std::size_t testSize = 8192) {
    BL_INFO("Allocating CUDA memory...");
    static I8* input_ptr;
    static I8* output_ptr;

    BL_CUDA_CHECK(cudaMallocManaged(&input_ptr, testSize * sizeof(I8)), [&]{
        BL_FATAL("Can't allocate complex checker test input buffer: {}", err);
    });

    BL_CUDA_CHECK(cudaMallocManaged(&output_ptr, testSize * sizeof(I8)), [&]{
        BL_FATAL("Can't allocate complex checker test output buffer: {}", err);
    });

    Vector<Device::CPU, I8> input{input_ptr, testSize};
    Vector<Device::CPU, I8> output{output_ptr, testSize};

    BL_INFO("Generating test data...");
    std::srand(unsigned(std::time(nullptr)));

    for (auto& element : input.getUnderlying()) {
        element = std::rand();
    }

    for (auto& element : output.getUnderlying()) {
        element = 60;
    }

    BL_INFO("Running kernels...");
    size_t counter = 0;

    if ((counter = Checker::run(input, output)) < testSize - 100) {
        BL_FATAL("[SUBTEST {}] Expected over than {} matches but found {}.",
                0, testSize - 100, counter);
        return Result::ERROR;
    }

    if ((counter = Checker::run(input, input)) > 100) {
        BL_FATAL("[SUBTEST {}] Expected less than {} matches but found {}.",
                1, 100, counter);
        return Result::ERROR;
    }

    if ((counter = Checker::run(output, output)) > 100) {
        BL_FATAL("[SUBTEST {}] Expected less than {} matches but found {}.",
                2, 100, counter);
        return Result::ERROR;
    }

    cudaFree(input_ptr);
    cudaFree(output_ptr);

    return Result::SUCCESS;
}

int main() {
    Logger guard{};

    BL_INFO("Testing integer checker.");

    if (Init() != Result::SUCCESS) {
        BL_WARN("Fault was encountered. Test is exiting...");
        return 1;
    }


    BL_INFO("Test succeeded.");

    return 0;
}
