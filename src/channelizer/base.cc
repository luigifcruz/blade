#include "blade/channelizer/base.hh"

#include "channelizer.jit.hh"

namespace Blade::Channelizer {

Generic::Generic(const Config & config) :
    Kernel(config.blockSize),
    config(config),
    cache(100, *channelizer_kernel)
{
    BL_DEBUG("Initilizating class.");

    if ((config.NTIME % config.fftSize) != 0) {
        BL_FATAL("The number of time samples ({}) should be divisable by the FFT size ({}).",
                config.NTIME, config.fftSize);
        throw Result::ERROR;
    }

    if (config.NBEAMS != 1) {
        BL_WARN("Number of beams ({}) should be one.", config.NBEAMS);
    }

    block = dim3(config.blockSize);
    grid = dim3((getBufferSize() + block.x - 1) / block.x / (config.fftSize * config.NPOLS));

    std::string kernel_key;
    switch (config.fftSize) {
        case 4: kernel_key = "fft_4pnt"; break;
        default:
            BL_FATAL("The FFT size of {} is not supported yet.", config.fftSize);
            throw Result::ERROR;
    }
    kernel = Template(kernel_key).instantiate(getBufferSize(), config.fftSize, config.NPOLS);
}

Generic::~Generic() {
    BL_DEBUG("Destroying class.");
}

Result Generic::run(const std::complex<int8_t>* input, std::complex<int8_t>* output) {
    cache.get_kernel(kernel)
        ->configure(grid, block)
        ->launch(
            reinterpret_cast<const char2*>(input),
            reinterpret_cast<char2*>(output)
        );

    BL_CUDA_CHECK_KERNEL([&]{
        BL_FATAL("Kernel failed to execute: {}", err);
        return Result::CUDA_ERROR;
    });

    return Result::SUCCESS;
}

} // namespace Blade::Channelizer
