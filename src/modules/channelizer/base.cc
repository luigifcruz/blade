#include "blade/modules/channelizer.hh"

#include "channelizer.jit.hh"

namespace Blade::Modules {

template<typename IT, typename OT>
Channelizer<IT, OT>::Channelizer(const Config& config, const Input& input)
        : Module(config.blockSize, channelizer_kernel),
          config(config),
          input(input) {
    if ((config.dims.NTIME % config.fftSize) != 0) {
        BL_FATAL("The number of time samples ({}) should be divisable "
                "by the FFT size ({}).", config.dims.NTIME, config.fftSize);
        throw Result::ERROR;
    }

    if (config.dims.NBEAMS != 1) {
        BL_WARN("Number of beams ({}) should be one.", config.dims.NBEAMS);
    }

    std::string kernel_key;
    switch (config.fftSize) {
        case 4: kernel_key = "fft_4pnt"; break;
        default:
            BL_FATAL("The FFT size of {} is not supported yet.", config.fftSize);
            throw Result::ERROR;
    }

    auto size = getBufferSize();

    grid = ((size / config.fftSize / config.dims.NPOLS) + block.x - 1) / block.x;
    kernel =
        Template(kernel_key)
            .instantiate(size, config.fftSize, config.dims.NPOLS,
                         config.dims.NTIME, config.dims.NCHANS);

    BL_CHECK_THROW(InitInput(input.buf, getBufferSize()));
    BL_CHECK_THROW(InitOutput(output.buf, getBufferSize()));
}

template<typename IT, typename OT>
Result Channelizer<IT, OT>::process(const cudaStream_t& stream) {
    cache
        .get_kernel(kernel)
        ->configure(grid, block, 0, stream)
        ->launch(input.buf.data(), output.buf.data());

    BL_CUDA_CHECK_KERNEL([&]{
        BL_FATAL("Module failed to execute: {}", err);
        return Result::CUDA_ERROR;
    });

    return Result::SUCCESS;
}

template class Channelizer<CF32, CF32>;

}  // namespace Blade::Modules
