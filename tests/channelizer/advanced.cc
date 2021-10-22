#include "blade/channelizer/test.hh"
#include "blade/channelizer/base.hh"
#include "blade/checker/base.hh"
#include "blade/manager.hh"

using namespace Blade;

Result Run(Channelizer::Generic & channelizer, Channelizer::Test::Generic & test) {
    Manager manager;
    Checker::Generic checker({});

    std::complex<float>* input_ptr;
    std::complex<float>* output_ptr;
    std::complex<float>* result_ptr;
    std::size_t buffer_size = channelizer.getBufferSize() * sizeof(input_ptr[0]);

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
    BL_CUDA_CHECK(cudaMallocManaged(&input_ptr, buffer_size), [&]{
        BL_FATAL("Can't allocate complex checker input buffer: {}", err);
    });

    BL_CUDA_CHECK(cudaMallocManaged(&output_ptr, buffer_size), [&]{
        BL_FATAL("Can't allocate complex checker output buffer: {}", err);
    });

    BL_CUDA_CHECK(cudaMallocManaged(&result_ptr, buffer_size), [&]{
        BL_FATAL("Can't allocate complex checker input buffer: {}", err);
    });

    std::span input(input_ptr, channelizer.getBufferSize());
    std::span output(output_ptr, channelizer.getBufferSize());
    std::span result(result_ptr, channelizer.getBufferSize());

    BL_INFO("Generating test data with Python...");
    BL_CHECK(test.process());

    BL_INFO("Checking test data size...");
    BL_ASSERT(test.getInputData().size() == channelizer.getBufferSize());
    BL_ASSERT(test.getOutputData().size() == channelizer.getBufferSize());

    BL_INFO("Copying test data to the device...");
    BL_CUDA_CHECK(cudaMemcpy(input.data(), test.getInputData().data(),
                buffer_size, cudaMemcpyHostToDevice), [&]{
        BL_FATAL("Can't copy beamformer input data from host to device: {}", err);
    });

    BL_CUDA_CHECK(cudaMemcpy(result.data(), test.getOutputData().data(),
                buffer_size, cudaMemcpyHostToDevice), [&]{
        BL_FATAL("Can't copy beamformer result data from host to device: {}", err);
    });

    BL_INFO("Running kernel...");
    for (int i = 0; i < 150; i++) {
        BL_CHECK(channelizer.run(input, output));
        cudaDeviceSynchronize();
    }

    BL_INFO("Checking for errors...");
    size_t errors = 0;
    if ((errors = checker.run(output, result)) != 0) {
        BL_FATAL("Beamformer produced {} errors.", errors);
        return Result::ERROR;
    }

    cudaFree(input_ptr);
    cudaFree(result_ptr);

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

    Channelizer::Test::Generic test(chan.getConfig());

    if (Run(chan, test) != Result::SUCCESS) {
        BL_WARN("Fault was encountered. Test is exiting...");
        return 1;
    }

    BL_INFO("Test succeeded.");

    return 0;
}
