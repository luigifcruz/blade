#include <blade/channelizer/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <jetstream/domains/dsp/invert/module.hh>
#include <jetstream/domains/core/reshape/module.hh>
#include <jetstream/domains/dsp/fft/module.hh>
#include <jetstream/memory/axis.hh>
#include <jetstream/tools/numeric.hh>

#include <optional>
#include <string>
#include <utility>

namespace Jetstream::Blocks {

struct ChannelizerImpl : public Block::Impl,
                         public DynamicConfig<Blocks::Channelizer> {
    Result validate() override;
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    struct CandidatePlan {
        std::string reshapeShape;
        SignalAxes signalAxes;
        U64 sampleDecimation = 1;
    };

    std::optional<CandidatePlan> candidatePlan;
    std::shared_ptr<Modules::Invert> invertConfig =
        std::make_shared<Modules::Invert>();
    std::shared_ptr<Modules::Fft> fftConfig = std::make_shared<Modules::Fft>();
    std::shared_ptr<Modules::Reshape> reshapeConfig =
        std::make_shared<Modules::Reshape>();
};

Result ChannelizerImpl::validate() {
    candidatePlan.reset();

    const auto input = inputs().find("buffer");
    if (input == inputs().end() || !input->second.resolved()) {
        return Result::SUCCESS;
    }

    const Tensor& inputTensor = input->second.tensor;
    if (!inputTensor.validShape() || inputTensor.size() == 0) {
        return Result::SUCCESS;
    }

    if (inputTensor.rank() != 4) {
        JST_ERROR("[BLOCK_CHANNELIZER] Expected rank-4 input [A,F,T,P], received shape {}.",
                  inputTensor.shape());
        return Result::ERROR;
    }

    if (inputTensor.dtype() != DataType::CF32) {
        JST_ERROR("[BLOCK_CHANNELIZER] Expected CF32 input, received {}.",
                  inputTensor.dtype());
        return Result::ERROR;
    }

    constexpr Index kAspectAxis = 0;
    constexpr Index kChannelAxis = 1;
    constexpr Index kSampleAxis = 2;
    constexpr Index kPolarizationAxis = 3;

    const U64 polarizations = inputTensor.shape(kPolarizationAxis);
    if (polarizations != 1 && polarizations != 2) {
        JST_ERROR("[BLOCK_CHANNELIZER] Expected one or two polarizations, received {}.",
                  polarizations);
        return Result::ERROR;
    }

    const U64 samples = inputTensor.shape(kSampleAxis);
    if (samples != 1 && (samples % 2) != 0) {
        JST_ERROR("[BLOCK_CHANNELIZER] Sample count {} must be one or even.", samples);
        return Result::ERROR;
    }

    SignalAxes inputAxes;
    if (MapSignalAxes(inputTensor,
                      IdentityAxisMap(inputTensor.rank()),
                      inputAxes) != Result::SUCCESS) {
        JST_ERROR("[BLOCK_CHANNELIZER] Input signal axis metadata is invalid.");
        return Result::ERROR;
    }
    if (inputAxes.sample && *inputAxes.sample != kSampleAxis) {
        JST_ERROR("[BLOCK_CHANNELIZER] sampleAxis must be absent or {}, but received {}.",
                  kSampleAxis,
                  *inputAxes.sample);
        return Result::ERROR;
    }
    if (inputAxes.channel && *inputAxes.channel != kChannelAxis) {
        JST_ERROR("[BLOCK_CHANNELIZER] channelAxis must be absent or {}, but received {}.",
                  kChannelAxis,
                  *inputAxes.channel);
        return Result::ERROR;
    }
    if (inputAxes.batch && *inputAxes.batch != kAspectAxis) {
        JST_ERROR("[BLOCK_CHANNELIZER] batchAxis must be absent or {}, but received {}.",
                  kAspectAxis,
                  *inputAxes.batch);
        return Result::ERROR;
    }

    const U64 aspects = inputTensor.shape(kAspectAxis);
    const U64 coarseChannels = inputTensor.shape(kChannelAxis);
    U64 fineChannels = 0;
    if (!detail::CheckedMultiply(coarseChannels, samples, fineChannels)) {
        JST_ERROR("[BLOCK_CHANNELIZER] Output channel count overflows U64.");
        return Result::ERROR;
    }

    const Shape outputShape = {aspects, fineChannels, 1, polarizations};
    U64 outputElementCount = 1;
    for (const U64 dimension : outputShape) {
        if (!detail::CheckedMultiply(outputElementCount,
                                     dimension,
                                     outputElementCount)) {
            JST_ERROR("[BLOCK_CHANNELIZER] Output shape exceeds the supported range.");
            return Result::ERROR;
        }
    }

    U64 outputSizeBytes = 0;
    if (!detail::CheckedMultiply(outputElementCount,
                                 static_cast<U64>(DataTypeSize(DataType::CF32)),
                                 outputSizeBytes)) {
        JST_ERROR("[BLOCK_CHANNELIZER] Output exceeds the supported byte range.");
        return Result::ERROR;
    }

    CandidatePlan plan;
    plan.reshapeShape =
        "[" + std::to_string(aspects) + "," + std::to_string(fineChannels) +
        ",1," + std::to_string(polarizations) + "]";
    plan.signalAxes = inputAxes;
    plan.signalAxes.sample = kSampleAxis;
    plan.signalAxes.channel = kChannelAxis;
    plan.sampleDecimation = samples;
    candidatePlan = std::move(plan);

    return Result::SUCCESS;
}

Result ChannelizerImpl::configure() {
    fftConfig->forward = true;

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
    if (!candidatePlan) {
        JST_ERROR("[BLOCK_CHANNELIZER] Input validation plan is unavailable.");
        return Result::ERROR;
    }

    TensorLink annotatedInput = inputPort;
    annotatedInput.tensor = inputPort.tensor.clone();
    JST_CHECK(SetSignalAxes(annotatedInput.tensor, candidatePlan->signalAxes));
    reshapeConfig->shape = candidatePlan->reshapeShape;

    JST_CHECK(moduleCreate("invert", invertConfig, {
        {"signal", annotatedInput}
    }));

    JST_CHECK(moduleCreate("fft", fftConfig, {
        {"signal", moduleGetOutput({"invert", "signal"})}
    }));
    JST_CHECK(moduleCreate("reshape", reshapeConfig, {
        {"buffer", moduleGetOutput({"fft", "signal"})}
    }));

    JST_CHECK(moduleExposeOutput("buffer", {"reshape", "buffer"}));
    JST_CHECK(SetSignalAxes(outputs().at("buffer").tensor,
                            candidatePlan->signalAxes));

    if (inputPort.tensor.hasAttribute("sampleRate")) {
        const Tensor inputCopy = inputPort.tensor;
        const F64 decimation = static_cast<F64>(candidatePlan->sampleDecimation);
        JST_CHECK(outputs().at("buffer").tensor.setDerivedAttribute(
            "sampleRate",
            [inputCopy, decimation]() -> std::any {
                const std::any sampleRate = inputCopy.attribute("sampleRate");
                if (const auto* value = std::any_cast<F32>(&sampleRate)) {
                    return std::any(static_cast<F32>(*value / decimation));
                }
                if (const auto* value = std::any_cast<F64>(&sampleRate)) {
                    return std::any(*value / decimation);
                }
                return {};
            }));
    }

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(ChannelizerImpl, {"invert"}, {"fft"}, {"reshape"});

}  // namespace Jetstream::Blocks
