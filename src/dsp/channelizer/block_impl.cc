#include <blade/channelizer/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <jetstream/domains/core/reshape/module.hh>
#include <jetstream/domains/dsp/fft/module.hh>

#include <limits>
#include <string>

namespace Jetstream::Blocks {

struct ChannelizerImpl : public Block::Impl,
                         public DynamicConfig<Blocks::Channelizer> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::Fft> fftConfig = std::make_shared<Modules::Fft>();
    std::shared_ptr<Modules::Reshape> reshapeConfig =
        std::make_shared<Modules::Reshape>();
};

Result ChannelizerImpl::configure() {
    fftConfig->forward = true;
    fftConfig->axis = 2;
    fftConfig->invert = true;

    return Result::SUCCESS;
}

Result ChannelizerImpl::define() {
    JST_CHECK(defineInterfaceInput(
        "buffer",
        "Input",
        "Contiguous CF32 voltages shaped [aspects, coarse channels, samples, polarizations]."));
    JST_CHECK(defineInterfaceOutput(
        "buffer",
        "Output",
        "Channelized CF32 voltages shaped [aspects, fine channels, spectra, polarizations]."));

    return Result::SUCCESS;
}

Result ChannelizerImpl::create() {
    const auto& inputPort = inputs().at("buffer");
    const Tensor& input = inputPort.tensor;

    if (input.rank() != 4) {
        JST_ERROR("[BLOCK_CHANNELIZER] Expected rank-4 input [A,F,T,P], received rank {}.",
                  input.rank());
        return Result::ERROR;
    }

    if (input.dtype() != DataType::CF32) {
        JST_ERROR("[BLOCK_CHANNELIZER] Expected CF32 input, received {}.", input.dtype());
        return Result::ERROR;
    }

    if (!input.contiguous()) {
        JST_ERROR("[BLOCK_CHANNELIZER] Expected contiguous input.");
        return Result::ERROR;
    }

    const U64 aspects = input.shape(0);
    const U64 coarseChannels = input.shape(1);
    const U64 samples = input.shape(2);
    const U64 polarizations = input.shape(3);

    if (aspects == 0 || coarseChannels == 0 || samples == 0) {
        JST_ERROR("[BLOCK_CHANNELIZER] Input dimensions must be greater than zero.");
        return Result::ERROR;
    }

    if (polarizations != 1 && polarizations != 2) {
        JST_ERROR("[BLOCK_CHANNELIZER] Expected one or two polarizations, received {}.",
                  polarizations);
        return Result::ERROR;
    }

    if (samples != 1 && (samples % 2) != 0) {
        JST_ERROR("[BLOCK_CHANNELIZER] Sample count {} must be even.", samples);
        return Result::ERROR;
    }

    if (coarseChannels > (std::numeric_limits<U64>::max() / samples)) {
        JST_ERROR("[BLOCK_CHANNELIZER] Output channel count overflows U64.");
        return Result::ERROR;
    }

    const U64 fineChannels = coarseChannels * samples;
    reshapeConfig->shape =
        "[" + std::to_string(aspects) + "," + std::to_string(fineChannels) + "," +
        "1," + std::to_string(polarizations) + "]";

    JST_CHECK(moduleCreate("fft", fftConfig, {
        {"signal", inputPort}
    }));
    JST_CHECK(moduleCreate("reshape", reshapeConfig, {
        {"buffer", moduleGetOutput({"fft", "signal"})}
    }));

    JST_CHECK(moduleExposeOutput("buffer", {"reshape", "buffer"}));

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(ChannelizerImpl);

}  // namespace Jetstream::Blocks
