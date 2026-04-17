#include "module_impl.hh"

namespace Jetstream::Modules {

Result BeamformerImpl::validate() {
    const auto& config = *candidate();

    if (config.blockSize == 0) {
        JST_ERROR("[MODULE_BEAMFORMER] The CUDA block size must be positive.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result BeamformerImpl::define() {
    JST_CHECK(defineInterfaceInput("buffer"));
    JST_CHECK(defineInterfaceInput("phasors"));
    JST_CHECK(defineInterfaceOutput("buffer"));

    return Result::SUCCESS;
}

Result BeamformerImpl::create() {
    inputTensor = inputs().at("buffer").tensor;
    phasorTensor = inputs().at("phasors").tensor;

    if (inputTensor.rank() != kBufferRank) {
        JST_ERROR("[MODULE_BEAMFORMER] Input tensor must have {} dimensions [A, F, T, P], but received shape {}.",
                  kBufferRank,
                  inputTensor.shape());
        return Result::ERROR;
    }

    if (phasorTensor.rank() != kPhasorRank) {
        JST_ERROR("[MODULE_BEAMFORMER] Phasor tensor must have {} dimensions [B, A, F, T, P], but received shape {}.",
                  kPhasorRank,
                  phasorTensor.shape());
        return Result::ERROR;
    }

    if (!inputTensor.contiguous()) {
        JST_ERROR("[MODULE_BEAMFORMER] Input tensor must be contiguous.");
        return Result::ERROR;
    }

    if (!phasorTensor.contiguous()) {
        JST_ERROR("[MODULE_BEAMFORMER] Phasor tensor must be contiguous.");
        return Result::ERROR;
    }

    if (inputTensor.shape(kBufferAspectAxis) == 0 ||
        inputTensor.shape(kBufferFrequencyAxis) == 0 ||
        inputTensor.shape(kBufferTimeAxis) == 0) {
        JST_ERROR("[MODULE_BEAMFORMER] Input tensor dimensions must be positive, but received shape {}.",
                  inputTensor.shape());
        return Result::ERROR;
    }

    if (phasorTensor.shape(kPhasorBeamAxis) == 0) {
        JST_ERROR("[MODULE_BEAMFORMER] Phasor tensor must contain at least one beam.");
        return Result::ERROR;
    }

    if (inputTensor.shape(kBufferPolarizationAxis) != kExpectedPolarizations) {
        JST_ERROR("[MODULE_BEAMFORMER] Input polarization dimension must be {}, but received {}.",
                  kExpectedPolarizations,
                  inputTensor.shape(kBufferPolarizationAxis));
        return Result::ERROR;
    }

    if (phasorTensor.shape(kPhasorPolarizationAxis) != kExpectedPolarizations) {
        JST_ERROR("[MODULE_BEAMFORMER] Phasor polarization dimension must be {}, but received {}.",
                  kExpectedPolarizations,
                  phasorTensor.shape(kPhasorPolarizationAxis));
        return Result::ERROR;
    }

    if (phasorTensor.shape(kPhasorAspectAxis) != inputTensor.shape(kBufferAspectAxis)) {
        JST_ERROR("[MODULE_BEAMFORMER] Number of antennas mismatch between phasors ({}) and input ({}).",
                  phasorTensor.shape(kPhasorAspectAxis),
                  inputTensor.shape(kBufferAspectAxis));
        return Result::ERROR;
    }

    if (phasorTensor.shape(kPhasorFrequencyAxis) != inputTensor.shape(kBufferFrequencyAxis)) {
        JST_ERROR("[MODULE_BEAMFORMER] Number of frequency channels mismatch between phasors ({}) and input ({}).",
                  phasorTensor.shape(kPhasorFrequencyAxis),
                  inputTensor.shape(kBufferFrequencyAxis));
        return Result::ERROR;
    }

    if (phasorTensor.shape(kPhasorTimeAxis) != kExpectedPhasorTimeSamples) {
        JST_ERROR("[MODULE_BEAMFORMER] Phasor time dimension must be {}, but received {}.",
                  kExpectedPhasorTimeSamples,
                  phasorTensor.shape(kPhasorTimeAxis));
        return Result::ERROR;
    }

    if ((inputTensor.shape(kBufferTimeAxis) % blockSize) != 0) {
        JST_ERROR("[MODULE_BEAMFORMER] Number of time samples ({}) isn't divisible by the block size ({}).",
                  inputTensor.shape(kBufferTimeAxis),
                  blockSize);
        return Result::ERROR;
    }

    if (phasorTensor.shape(kPhasorBeamAxis) > blockSize) {
        JST_ERROR("[MODULE_BEAMFORMER] The block size ({}) is smaller than the number of beams ({}).",
                  blockSize,
                  phasorTensor.shape(kPhasorBeamAxis));
        return Result::ERROR;
    }

    beamCount = phasorTensor.shape(kPhasorBeamAxis);

    const Shape outputShape = {
        beamCount + U64(enableIncoherentBeam ? 1 : 0),
        inputTensor.shape(kBufferFrequencyAxis),
        inputTensor.shape(kBufferTimeAxis),
        inputTensor.shape(kBufferPolarizationAxis),
    };

    JST_CHECK(outputTensor.create(inputTensor.device(), DataType::CF32, outputShape));
    JST_CHECK(outputTensor.propagateAttributes(inputTensor));

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
