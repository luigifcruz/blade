#define BL_LOG_DOMAIN "M::BEAMFORMER"

#include "blade/modules/beamformer/generic.hh"

#include "beamformer.jit.hh"

namespace Blade::Modules::Beamformer {

template<typename IT, typename OT>
Generic<IT, OT>::Generic(const Config& config, const Input& input)
        : Module(beamformer_program),
          config(config),
          input(input) {
    // Check configuration values.
    if ((getInputBuffer().dims().numberOfTimeSamples() % config.blockSize) != 0) {
        BL_FATAL("Number of time samples ({}) isn't divisable by the block size ({}).", 
                getInputBuffer().dims().numberOfTimeSamples(), config.blockSize);
        BL_CHECK_THROW(Result::ERROR);
    }

    // Print configuration values.
    BL_INFO("Type: {} -> {}", TypeInfo<IT>::name, TypeInfo<OT>::name);
    BL_INFO("Phasors Dimensions [B, A, F, T, P]: {}", getInputPhasors().dims());
    BL_INFO("Enable Incoherent Beam: {}", config.enableIncoherentBeam ? "YES" : "NO");
    BL_INFO("Enable Incoherent Beam Square Root: {}", config.enableIncoherentBeamSqrt ? "YES" : "NO");
}

template<typename IT, typename OT>
const Result Generic<IT, OT>::process(const cudaStream_t& stream) {
    return runKernel("main", stream, input.buf.data(), input.phasors.data(), output.buf.data());
}

template class BLADE_API Generic<CF32, CF32>;

}  // namespace Blade::Modules::Beamformer
