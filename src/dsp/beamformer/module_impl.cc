#include "module_impl.hh"

#include <jetstream/tools/numeric.hh>

namespace Jetstream::Modules {

Result BeamformerImpl::validate() {
    validatedBeamCount = 0;
    validatedOutputSizeBytes = 0;
    validatedOutputShape.clear();
    validatedSignalAxes = {};

    const auto& config = *candidate();

    if (!inputs().contains("buffer") || !inputs().contains("phasors")) {
        return Result::SUCCESS;
    }

    const Tensor& input = inputs().at("buffer").tensor;
    const Tensor& phasors = inputs().at("phasors").tensor;
    if (!input.validShape() || input.size() == 0 ||
        !phasors.validShape() || phasors.size() == 0) {
        return Result::SUCCESS;
    }

    if (input.rank() != kBufferRank) {
        JST_ERROR("[MODULE_BEAMFORMER] Input tensor must have {} dimensions [A, F, T, P], but received shape {}.",
                  kBufferRank,
                  input.shape());
        return Result::ERROR;
    }

    if (phasors.rank() != kPhasorRank) {
        JST_ERROR("[MODULE_BEAMFORMER] Phasor tensor must have {} dimensions [B, A, F, T, P], but received shape {}.",
                  kPhasorRank,
                  phasors.shape());
        return Result::ERROR;
    }

    SignalAxes axes;
    if (MapSignalAxes(input,
                      IdentityAxisMap(input.rank()),
                      axes) != Result::SUCCESS) {
        JST_ERROR("[MODULE_BEAMFORMER] Input must contain valid signal axis metadata.");
        return Result::ERROR;
    }

    if (axes.sample && *axes.sample != kBufferTimeAxis) {
        JST_ERROR("[MODULE_BEAMFORMER] sampleAxis must be absent or {}, but received {}.",
                  kBufferTimeAxis,
                  *axes.sample);
        return Result::ERROR;
    }

    if (axes.channel && *axes.channel != kBufferFrequencyAxis) {
        JST_ERROR("[MODULE_BEAMFORMER] channelAxis must be absent or {}, but received {}.",
                  kBufferFrequencyAxis,
                  *axes.channel);
        return Result::ERROR;
    }
    if (axes.batch && *axes.batch != kBufferAspectAxis) {
        JST_ERROR("[MODULE_BEAMFORMER] batchAxis must be absent or {}, but received {}.",
                  kBufferAspectAxis,
                  *axes.batch);
        return Result::ERROR;
    }

    axes.sample = kBufferTimeAxis;
    axes.channel = kBufferFrequencyAxis;

    if (input.shape(kBufferPolarizationAxis) != kExpectedPolarizations) {
        JST_ERROR("[MODULE_BEAMFORMER] Input polarization dimension must be {}, but received {}.",
                  kExpectedPolarizations,
                  input.shape(kBufferPolarizationAxis));
        return Result::ERROR;
    }

    if (phasors.shape(kPhasorPolarizationAxis) != kExpectedPolarizations) {
        JST_ERROR("[MODULE_BEAMFORMER] Phasor polarization dimension must be {}, but received {}.",
                  kExpectedPolarizations,
                  phasors.shape(kPhasorPolarizationAxis));
        return Result::ERROR;
    }

    if (phasors.shape(kPhasorAspectAxis) != input.shape(kBufferAspectAxis)) {
        JST_ERROR("[MODULE_BEAMFORMER] Number of antennas mismatch between phasors ({}) and input ({}).",
                  phasors.shape(kPhasorAspectAxis),
                  input.shape(kBufferAspectAxis));
        return Result::ERROR;
    }

    if (phasors.shape(kPhasorFrequencyAxis) != input.shape(kBufferFrequencyAxis)) {
        JST_ERROR("[MODULE_BEAMFORMER] Number of frequency channels mismatch between phasors ({}) and input ({}).",
                  phasors.shape(kPhasorFrequencyAxis),
                  input.shape(kBufferFrequencyAxis));
        return Result::ERROR;
    }

    if (phasors.shape(kPhasorTimeAxis) != kExpectedPhasorTimeSamples) {
        JST_ERROR("[MODULE_BEAMFORMER] Phasor time dimension must be {}, but received {}.",
                  kExpectedPhasorTimeSamples,
                  phasors.shape(kPhasorTimeAxis));
        return Result::ERROR;
    }

    const U64 candidateBeamCount = phasors.shape(kPhasorBeamAxis);
    U64 outputBeamCount = 0;
    if (!detail::CheckedAdd(candidateBeamCount,
                            config.enableIncoherentBeam ? U64{1} : U64{0},
                            outputBeamCount)) {
        JST_ERROR("[MODULE_BEAMFORMER] Output beam count exceeds the supported range.");
        return Result::ERROR;
    }

    Shape outputShape = {
        outputBeamCount,
        input.shape(kBufferFrequencyAxis),
        input.shape(kBufferTimeAxis),
        input.shape(kBufferPolarizationAxis),
    };

    U64 outputElementCount = 1;
    for (const U64 dimension : outputShape) {
        if (!detail::CheckedMultiply(outputElementCount,
                                     dimension,
                                     outputElementCount)) {
            JST_ERROR("[MODULE_BEAMFORMER] Output shape exceeds the supported range.");
            return Result::ERROR;
        }
    }

    U64 outputSizeBytes = 0;
    if (!detail::CheckedMultiply(outputElementCount,
                                 static_cast<U64>(DataTypeSize(DataType::CF32)),
                                 outputSizeBytes)) {
        JST_ERROR("[MODULE_BEAMFORMER] Output exceeds the supported byte range.");
        return Result::ERROR;
    }

    validatedBeamCount = candidateBeamCount;
    validatedOutputSizeBytes = outputSizeBytes;
    validatedOutputShape = std::move(outputShape);
    validatedSignalAxes = axes;

    return Result::SUCCESS;
}

Result BeamformerImpl::define() {
    JST_CHECK(defineTaint(Module::Taint::STATELESS));

    JST_CHECK(defineInterfaceInput("buffer"));
    JST_CHECK(defineInterfaceInput("phasors"));
    JST_CHECK(defineInterfaceOutput("buffer"));

    return Result::SUCCESS;
}

Result BeamformerImpl::create() {
    inputTensor = inputs().at("buffer").tensor;
    phasorTensor = inputs().at("phasors").tensor;
    beamCount = validatedBeamCount;

    JST_CHECK(outputTensor.create(inputTensor.device(), DataType::CF32, validatedOutputShape));
    JST_CHECK(outputTensor.propagateAttributes(inputTensor));
    JST_CHECK(SetSignalAxes(outputTensor, validatedSignalAxes));

    outputs()["buffer"].produced(name(), "buffer", outputTensor);

    return Result::SUCCESS;
}

Result BeamformerImpl::destroy() {
    inputTensor = {};
    phasorTensor = {};
    outputTensor = {};
    beamCount = 0;

    return Result::SUCCESS;
}

Result BeamformerImpl::reconfigure() {
    return Result::RECREATE;
}

}  // namespace Jetstream::Modules
