#include "bl-beamformer/base.hh"

using namespace BL;

Result Init() {
    Beamformer beam({
        .NBEAMS = 16,
        .NANTS  = 20,
        .NCHANS = 384,
        .NTIME  = 8750,
        .NPOLS  = 2,
        .TBLOCK = 350,
        .kernel = Beamformer::Kernel::ATA,
    });

    Checker checker({beam.outputLen()});

    std::complex<int8_t>* input;
    BL_CUDA_CHECK(cudaMalloc(&input, beam.inputLen() * sizeof(std::complex<int8_t>)), [&]{
        BL_FATAL("Can't allocate beamformer input buffer.");
    });

    std::complex<float>* phasor;
    BL_CUDA_CHECK(cudaMalloc(&phasor, beam.phasorLen() * sizeof(std::complex<float>)), [&]{
        BL_FATAL("Can't allocate beamformer phasor buffer.");
    });

    std::complex<float>* output;
    BL_CUDA_CHECK(cudaMalloc(&output, beam.outputLen() * sizeof(std::complex<float>)), [&]{
        BL_FATAL("Can't allocate beamformer output buffer.");
    });

    std::complex<float>* result;
    BL_CUDA_CHECK(cudaMalloc(&result, beam.outputLen() * sizeof(std::complex<float>)), [&]{
        BL_FATAL("Can't allocate beamformer output groundtruth buffer.");
    });

    BL_CHECK(Helpers::LoadFromFile("input.raw", input, sizeof(std::complex<int8_t>), beam.inputLen()));
    BL_CHECK(Helpers::LoadFromFile("phasor.raw", phasor, sizeof(std::complex<float>), beam.phasorLen()));
    BL_CHECK(Helpers::LoadFromFile("output.raw", result, sizeof(std::complex<float>), beam.outputLen()));

    for (int i = 0; i < 100; i++) {
        BL_CHECK(beam.run(input, phasor, output));
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

int main() {
    Logger::Init();

    BL_INFO("Welcome to BL Beamformer.");

    if (Init() != Result::SUCCESS) {
        BL_WARN("Fault was encountered. System is exiting...");
        return 1;
    }

    BL_INFO("Nominal shutdown...");
    Logger::Shutdown();

    return 0;
}
