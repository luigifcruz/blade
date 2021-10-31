#include "blade/beamformer/generic.hh"

#include "beamformer.jit.hh"

namespace Blade::Beamformer {

Generic::Generic(const Config& config) :
    Kernel(config.blockSize), config(config), cache(100, *beamformer_kernel) {
    BL_DEBUG("Initilizating class.");

    if ((config.dims.NTIME % config.blockSize) != 0) {
        BL_FATAL("Number of time samples ({}) isn't divisable by "
                "the block size ({}).", config.dims.NTIME, config.blockSize);
        throw Result::ERROR;
    }
}

Result Generic::run(const std::span<std::complex<F32>>& input,
                    const std::span<std::complex<F32>>& phasors,
                          std::span<std::complex<F32>>& output,
                          cudaStream_t cudaStream) {
    if (input.size() != getInputSize()) {
        BL_FATAL("Size mismatch between input and configuration ({}, {}).",
                input.size(), getInputSize());
        return Result::ASSERTION_ERROR;
    }

    if (phasors.size() != getPhasorsSize()) {
        BL_FATAL("Size mismatch between phasors and configuration ({}, {}).",
                phasors.size(), getPhasorsSize());
        return Result::ASSERTION_ERROR;
    }

    if (output.size() != getOutputSize()) {
        BL_FATAL("Size mismatch between output and configuration ({}, {}).",
                output.size(), getOutputSize());
        return Result::ASSERTION_ERROR;
    }

    cache.get_kernel(kernel)
        ->configure(grid, block, 0, cudaStream)
        ->launch(
            reinterpret_cast<const cuFloatComplex*>(input.data()),
            reinterpret_cast<const cuFloatComplex*>(phasors.data()),
            reinterpret_cast<cuFloatComplex*>(output.data()));

    BL_CUDA_CHECK_KERNEL([&]{
        BL_FATAL("Kernel failed to execute: {}", err);
        return Result::CUDA_ERROR;
    });

    return Result::SUCCESS;
}

}  // namespace Blade::Beamformer
