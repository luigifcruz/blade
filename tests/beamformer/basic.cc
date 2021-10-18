#include "blade/beamformer/generic.hh"
#include "blade/checker/base.hh"
#include "blade/manager.hh"

using namespace Blade;

Result Run(Beamformer::Generic & beam) {
    Manager manager;
    Checker::Generic checker({beam.getOutputSize()});

    std::complex<int8_t>* input;
    std::size_t input_size = beam.getInputSize() * sizeof(input[0]);

    std::complex<float>* phasors;
    std::size_t phasors_size = beam.getPhasorsSize() * sizeof(phasors[0]);

    std::complex<float>* output;
    std::size_t output_size = beam.getOutputSize() * sizeof(output[0]);

    std::complex<float>* result;
    std::size_t result_size = beam.getOutputSize() * sizeof(result[0]);

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
    BL_CUDA_CHECK(cudaMallocManaged(&input, input_size), [&]{
        BL_FATAL("Can't allocate beamformer input buffer: {}", err);
    });

    BL_CUDA_CHECK(cudaMallocManaged(&phasors, phasors_size), [&]{
        BL_FATAL("Can't allocate beamformer phasor buffer: {}", err);
    });

    BL_CUDA_CHECK(cudaMallocManaged(&output, output_size), [&]{
        BL_FATAL("Can't allocate beamformer output buffer: {}", err);
    });

    BL_CUDA_CHECK(cudaMallocManaged(&result, result_size), [&]{
        BL_FATAL("Can't allocate beamformer output groundtruth buffer: {}", err);
    });

    BL_INFO("Generating test data...");
    std::span<std::complex<int8_t>> input_span{input, beam.getInputSize()};
    std::generate(input_span.begin(), input_span.end(), []{ return 1; });

    std::span<std::complex<float>> phasor_span{phasors, beam.getPhasorsSize()};
    std::generate(phasor_span.begin(), phasor_span.end(), []{ return 2.0; });

    std::span<std::complex<float>> result_span{result, beam.getOutputSize()};
    std::generate(result_span.begin(), result_span.end(), [&]{ return beam.getConfig().NANTS * 2.0; });

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

    cudaFree(input);
    cudaFree(output);
    cudaFree(phasors);
    cudaFree(result);

    return Result::SUCCESS;
}
