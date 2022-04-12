#include "blade/modules/channelizer.hh"

#include "channelizer.jit.hh"

namespace Blade::Modules {

// TODO: Implement multiple beams capability;

template<typename IT, typename OT>
Channelizer<IT, OT>::Channelizer(const Config& config, const Input& input)
        : Module(config.blockSize, channelizer_kernel),
          config(config),
          input(input) {
    if ((config.numberOfTimeSamples % config.rate) != 0) {
        BL_FATAL("The number of time samples ({}) should be divisable "
                "by the channelizer rate ({}).", config.numberOfTimeSamples,
                config.rate);
        throw Result::ERROR;
    }

    if (config.numberOfBeams != 1) {
        BL_WARN("Number of beams ({}) should be one.", config.numberOfBeams);
    }

    if (config.rate == 1) {
        BL_CHECK_THROW(output.buf.link(input.buf));
        return;
    }

    std::string kernel_key;
    switch (config.rate) {
        case 4: kernel_key = "fft_4pnt"; break;
        default:
            BL_FATAL("The channelize rate of {} is not supported yet.", config.rate);
            throw Result::ERROR;
    }

    grid = 
        (
            (
                getBufferSize() / 
                config.rate /
                config.numberOfPolarizations
            ) + block.x - 1
        ) / block.x;

    kernel =
        Template(kernel_key)
            .instantiate(getBufferSize(),
                         config.rate,
                         config.numberOfPolarizations,
                         config.numberOfTimeSamples,
                         config.numberOfFrequencyChannels);

    BL_CHECK_THROW(InitInput(input.buf, getBufferSize()));
    BL_CHECK_THROW(InitOutput(output.buf, getBufferSize()));
}

template<typename IT, typename OT>
Result Channelizer<IT, OT>::process(const cudaStream_t& stream) {
    if (config.rate == 1) {
        return Result::SUCCESS;
    }

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

template class BLADE_API Channelizer<CF32, CF32>;

}  // namespace Blade::Modules
