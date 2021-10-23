#include "blade/cast/base.hh"
#include "blade/checker/base.hh"
#include "blade/manager.hh"

using namespace Blade;

template<typename IT, typename OT, typename FG>
Result Run(FG gen_func, std::size_t testSize = 134400000) {
    Checker::Generic checker({});
    Cast::Generic cast({});
    Manager manager;

    BL_INFO("Allocating CUDA memory...");
    static IT* input_ptr;
    std::size_t input_size = testSize * sizeof(IT);

    static OT* output_ptr;
    std::size_t output_size = testSize * sizeof(OT);

    static OT* result_ptr;
    std::size_t result_size = testSize * sizeof(OT);

    manager.save({
        .memory = {
            .device = input_size + output_size + result_size,
            .host = input_size + output_size + result_size,
        },
        .transfer = {
            .d2h = input_size,
            .h2d = result_size,
        }
    }).report();
    BL_CUDA_CHECK(cudaMallocManaged(&input_ptr, input_size), [&]{
        BL_FATAL("Can't allocate test input buffer: {}", err);
    });

    BL_CUDA_CHECK(cudaMallocManaged(&output_ptr, output_size), [&]{
        BL_FATAL("Can't allocate test output buffer: {}", err);
    });

    BL_CUDA_CHECK(cudaMallocManaged(&result_ptr, result_size), [&]{
        BL_FATAL("Can't allocate test result buffer: {}", err);
    });

    std::span input(input_ptr, testSize);
    std::span output(output_ptr, testSize);
    std::span result(result_ptr, testSize);

    BL_INFO("Generating test data...");
    gen_func(input, result);

    BL_INFO("Running kernels...");
    for (int i = 0; i < 150; i++) {
        BL_CHECK(cast.run(input, output));
        cudaDeviceSynchronize();
    }

    BL_INFO("Checking for errors...");
    size_t errors = 0;
    if ((errors = checker.run(result, output)) != 0) {
        BL_FATAL("Beamformer produced {} errors.", errors);
        return Result::ERROR;
    }

    cudaFree(input_ptr);
    cudaFree(output_ptr);
    cudaFree(result_ptr);

    return Result::SUCCESS;
}

int main() {
    Logger guard{};
    std::srand(unsigned(std::time(nullptr)));

    BL_INFO("Testing advanced channelizer.");

    BL_INFO("Testing complex int8 to complex float conversion...");
    if (Run<std::complex<int8_t>, std::complex<float>>([](auto &input, auto &result){
        for (std::size_t i = 0; i < input.size(); i++) {
            input[i] = {
                static_cast<int8_t>(std::rand()),
                static_cast<int8_t>(std::rand())
            };

            result[i] = {
                static_cast<float>(input[i].real()),
                static_cast<float>(input[i].imag())
            };
        }
    }) != Result::SUCCESS) {
        BL_WARN("Fault was encountered. Test is exiting...");
        return 1;
    }

    BL_INFO("Test succeeded.");

    return 0;
}
