#include <span>

#include "bl-beamformer/base.hh"

using namespace BL;

Result Run(const Beamformer::Config & config) {
    Beamformer beam(config);

    Checker checker({beam.outputLen()});

    std::complex<int8_t>* input;
    CUDA_CHECK(cudaMallocManaged(&input, beam.inputLen() * sizeof(std::complex<int8_t>)), [&]{
        BL_FATAL("Can't allocate beamformer input buffer.");
    });

    std::complex<float>* phasor;
    CUDA_CHECK(cudaMallocManaged(&phasor, beam.phasorLen() * sizeof(std::complex<float>)), [&]{
        BL_FATAL("Can't allocate beamformer phasor buffer.");
    });

    std::complex<float>* output;
    CUDA_CHECK(cudaMallocManaged(&output, beam.outputLen() * sizeof(std::complex<float>)), [&]{
        BL_FATAL("Can't allocate beamformer output buffer.");
    });

    std::complex<float>* result;
    CUDA_CHECK(cudaMallocManaged(&result, beam.outputLen() * sizeof(std::complex<float>)), [&]{
        BL_FATAL("Can't allocate beamformer output groundtruth buffer.");
    });

    std::span<std::complex<int8_t>> input_span{input, beam.inputLen()};
    std::generate(input_span.begin(), input_span.end(), []{ return 1; });

    std::span<std::complex<float>> phasor_span{phasor, beam.phasorLen()};
    std::generate(phasor_span.begin(), phasor_span.end(), []{ return 2.0; });

    std::span<std::complex<float>> result_span{result, beam.outputLen()};
    std::generate(result_span.begin(), result_span.end(), [&]{ return config.NANTS * 2.0; });

    for (int i = 0; i < 150; i++) {
        CHECK(beam.run(input, phasor, output));
        cudaDeviceSynchronize();
    }

    size_t errors = 0;
    if ((errors = checker.run(output, result)) != 0) {
        BL_FATAL("Beamformer produced {} errors.", errors);
        return Result::ERROR;
    }

    cudaFree(input);
    cudaFree(output);
    cudaFree(phasor);
    cudaFree(result);

    return Result::SUCCESS;
}
