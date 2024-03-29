#define BL_LOG_DOMAIN "M::BEAMFORMER"

#include "blade/modules/beamformer/generic.hh"

#include "beamformer.jit.hh"

namespace Blade::Modules::Beamformer {

template<typename IT, typename OT>
Generic<IT, OT>::Generic(const Config& config, 
                         const Input& input,
                         const Stream& stream)
        : Module(beamformer_program),
          config(config),
          input(input) {
    // Check configuration values.
    if ((getInputBuffer().shape().numberOfTimeSamples() % config.blockSize) != 0) {
        BL_FATAL("Number of time samples ({}) isn't divisable by the block size ({}).", 
                getInputBuffer().shape().numberOfTimeSamples(), config.blockSize);
        BL_CHECK_THROW(Result::ERROR);
    }

    // Print configuration values.
    BL_INFO("Type: {} -> {}", TypeInfo<IT>::name, TypeInfo<OT>::name);
    BL_INFO("Phasors Shape: {}", getInputPhasors().shape());
    BL_INFO("Enable Incoherent Beam: {}", config.enableIncoherentBeam ? "YES" : "NO");
    BL_INFO("Enable Incoherent Beam Square Root: {}", config.enableIncoherentBeamSqrt ? "YES" : "NO");
}

template<typename IT, typename OT>
Result Generic<IT, OT>::process(const U64& currentStepCount, const Stream& stream) {
    return runKernel("main", stream, input.buf.data(), input.phasors.data(), output.buf.data());
}

template class BLADE_API Generic<CF32, CF32>;

}  // namespace Blade::Modules::Beamformer
