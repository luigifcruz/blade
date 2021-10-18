#include "blade/channelizer/test.hh"
#include "blade/channelizer/base.hh"
#include "blade/checker/base.hh"
#include "blade/manager.hh"

using namespace Blade;

Result Run(Channelizer::Generic & channelizer, Channelizer::Test::Generic & test) {
    Manager manager;
    Checker::Generic checker({channelizer.getBufferSize()*2});

    std::complex<int8_t>* input;
    std::complex<int8_t>* output;
    std::complex<int8_t>* result;
    std::size_t buffer_size = channelizer.getBufferSize() * sizeof(input[0]);

    manager.save({
        .memory = {
            .device = buffer_size*3,
            .host = buffer_size*3,
        },
        .transfer = {
            .d2h = buffer_size,
            .h2d = buffer_size,
        }
    }).report();

    BL_INFO("Allocating CUDA memory...");
    BL_CUDA_CHECK(cudaMallocManaged(&input, buffer_size), [&]{
        BL_FATAL("Can't allocate complex checker input buffer: {}", err);
    });

    BL_CUDA_CHECK(cudaMallocManaged(&output, buffer_size), [&]{
        BL_FATAL("Can't allocate complex checker output buffer: {}", err);
    });

    BL_CUDA_CHECK(cudaMallocManaged(&result, buffer_size), [&]{
        BL_FATAL("Can't allocate complex checker input buffer: {}", err);
    });

    BL_INFO("Generating test data with Python...");
    BL_CHECK(test.process());

    BL_INFO("Checking test data size...");
    BL_ASSERT(test.getInputData().size() == channelizer.getBufferSize());
    BL_ASSERT(test.getOutputData().size() == channelizer.getBufferSize());

    BL_INFO("Copying test data to the device...");
    BL_CUDA_CHECK(cudaMemcpy(input, test.getInputData().data(), buffer_size, cudaMemcpyHostToDevice), [&]{
        BL_FATAL("Can't copy beamformer input data from host to device: {}", err);
    });

    BL_CUDA_CHECK(cudaMemcpy(result, test.getOutputData().data(), buffer_size, cudaMemcpyHostToDevice), [&]{
        BL_FATAL("Can't copy beamformer result data from host to device: {}", err);
    });

    BL_INFO("Running kernel...");
    for (int i = 0; i < 150; i++) {
        BL_CHECK(channelizer.run(input, output));
        cudaDeviceSynchronize();
    }

    BL_INFO("Checking for errors...");
    size_t errors = 0;
    if ((errors = checker.run(reinterpret_cast<int8_t*>(output), reinterpret_cast<int8_t*>(result))) != 0) {
        BL_FATAL("Beamformer produced {} errors.", errors);
        return Result::ERROR;
    }

    cudaFree(input);
    cudaFree(result);

    return Result::SUCCESS;
};

int main() {
    Logger guard{};

    BL_INFO("Testing advanced channelizer.");

    Channelizer::Generic chan({
        {
            .NBEAMS = 1,
            .NANTS  = 20,
            .NCHANS = 96,
            .NTIME  = 35000,
            .NPOLS  = 2,
        },
        4,
        1024,
    });

    Channelizer::Test::GenericPython test(chan.getConfig());

    if (Run(chan, test) != Result::SUCCESS) {
        BL_FATAL("Test failed.");
        return 1;
    }

    BL_INFO("Test succeeded.");

    return 0;
}
