#include "blade/beamformer/test/generic.hh"
#include "blade/beamformer/generic.hh"
#include "blade/checker/base.hh"
#include "blade/manager.hh"

using namespace Blade;

Result Run(Beamformer::Generic & beam, Beamformer::Test::Generic & test) {
    Manager manager;
    Checker::Generic checker({});

    std::complex<float>* input_ptr;
    std::size_t input_size = beam.getInputSize() * sizeof(input_ptr[0]);

    std::complex<float>* phasors_ptr;
    std::size_t phasors_size = beam.getPhasorsSize() * sizeof(phasors_ptr[0]);

    std::complex<float>* output_ptr;
    std::size_t output_size = beam.getOutputSize() * sizeof(output_ptr[0]);

    std::complex<float>* result_ptr;
    std::size_t result_size = beam.getOutputSize() * sizeof(result_ptr[0]);

    manager.save({
        .memory = {
            .device = input_size + phasors_size + output_size + result_size,
            .host = input_size + phasors_size + output_size + result_size,
        },
        .transfer = {
            .d2h = result_size,
            .h2d = input_size,
        }
    }).report();

    BL_INFO("Allocating CUDA memory...");
    BL_CUDA_CHECK(cudaMalloc(&input_ptr, input_size), [&]{
        BL_FATAL("Can't allocate beamformer input buffer: {}", err);
    });

    BL_CUDA_CHECK(cudaMalloc(&phasors_ptr, phasors_size), [&]{
        BL_FATAL("Can't allocate beamformer phasor buffer: {}", err);
    });

    BL_CUDA_CHECK(cudaMalloc(&output_ptr, output_size), [&]{
        BL_FATAL("Can't allocate beamformer output buffer: {}", err);
    });

    BL_CUDA_CHECK(cudaMalloc(&result_ptr, result_size), [&]{
        BL_FATAL("Can't allocate beamformer output groundtruth buffer: {}", err);
    });

    std::span input(input_ptr, beam.getInputSize());
    std::span output(output_ptr, beam.getOutputSize());
    std::span result(result_ptr, beam.getOutputSize());
    std::span phasors(phasors_ptr, beam.getPhasorsSize());

    BL_INFO("Generating test data with Python...");
    BL_CHECK(test.process());

    BL_INFO("Checking test data size...");
    BL_ASSERT(test.getInputData().size() == beam.getInputSize());
    BL_ASSERT(test.getPhasorsData().size() == beam.getPhasorsSize());
    BL_ASSERT(test.getOutputData().size() == beam.getOutputSize());

    BL_INFO("Copying test data to the device...");
    BL_CUDA_CHECK(cudaMemcpy(input.data(), test.getInputData().data(),
                input_size, cudaMemcpyHostToDevice), [&]{
        BL_FATAL("Can't copy beamformer input data from host to device: {}", err);
    });

    BL_CUDA_CHECK(cudaMemcpy(phasors.data(), test.getPhasorsData().data(),
                phasors_size, cudaMemcpyHostToDevice), [&]{
        BL_FATAL("Can't copy beamformer phasors data from host to device: {}", err);
    });

    BL_CUDA_CHECK(cudaMemcpy(result.data(), test.getOutputData().data(),
                result_size, cudaMemcpyHostToDevice), [&]{
        BL_FATAL("Can't copy beamformer result data from host to device: {}", err);
    });

    BL_INFO("Running kernel...");
    for (int i = 0; i < 150; i++) {
        BL_CHECK(beam.run(input, phasors, output));
        cudaDeviceSynchronize();
    }

    BL_INFO("Checking for errors...");
    size_t errors = 0;
    if ((errors = checker.run(output, result)) != 0) {
        BL_FATAL("Beamformer produced {} errors.", errors);
        return Result::ERROR;
    }

    cudaFree(input_ptr);
    cudaFree(output_ptr);
    cudaFree(phasors_ptr);
    cudaFree(result_ptr);

    return Result::SUCCESS;
}
