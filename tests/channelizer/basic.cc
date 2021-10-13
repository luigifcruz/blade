#include "blade/channelizer/base.hh"
#include "blade/checker/base.hh"
#include "blade/manager.hh"

using namespace Blade;

Result Run(Channelizer & channelizer) {
    Manager manager;
    Checker checker({channelizer.getInputSize()*2});

    std::complex<int8_t>* input;
    std::complex<int8_t>* result;
    std::size_t input_size = channelizer.getInputSize() * sizeof(input[0]);

    manager.save({
        .memory = {
            .device = input_size,
            .host = input_size,
        },
        .transfer = {
            .d2h = input_size,
            .h2d = input_size,
        }
    }).report();

    BL_CUDA_CHECK(cudaMallocManaged(&input, input_size), [&]{
        BL_FATAL("Can't allocate complex checker input buffer: {}", err);
    });

    BL_CUDA_CHECK(cudaMallocManaged(&result, input_size), [&]{
        BL_FATAL("Can't allocate complex checker input buffer: {}", err);
    });

    std::span<std::complex<int8_t>> result_span{result, channelizer.getInputSize()};
    std::generate(result_span.begin(), result_span.end(), []{ return std::complex<int8_t>(2,2); });

    for (int i = 0; i < 150; i++) {
        BL_CHECK(channelizer.run(input));
        cudaDeviceSynchronize();
    }

    size_t errors = 0;
    if ((errors = checker.run(reinterpret_cast<int8_t*>(input), reinterpret_cast<int8_t*>(result))) != 0) {
        BL_FATAL("Beamformer produced {} errors.", errors);
    }

    cudaFree(input);
    cudaFree(result);

    return Result::SUCCESS;
};

int main() {
    Logger guard{};

    BL_INFO("Testing beamformer with the ATA kernel.");

    Channelizer chan({
        {
            .NBEAMS = 1,
            .NANTS  = 20,
            .NCHANS = 96,
            .NTIME  = 35000,
            .NPOLS  = 2,
        },
        4,
        750,
    });

    if (Run(chan) != Result::SUCCESS) {
        BL_FATAL("Test failed.");
        return 1;
    }

    BL_INFO("Test succeeded.");

    return 0;
}

