#include "blade/modules/beamformer/meerkat.hh"

namespace Blade::Modules::Beamformer {

template<typename IT, typename OT>
MeerKAT<IT, OT>::MeerKAT(const typename Generic<IT, OT>::Config& config,
                         const typename Generic<IT, OT>::Input& input)
        : Generic<IT, OT>(config, input) {
    this->block = dim3(config.blockSize);
    this->grid = dim3(config.dims.NCHANS, config.dims.NTIME / config.blockSize);

    this->kernel = Template("MeerKAT")
        .instantiate(
            config.dims.NBEAMS,
            config.dims.NANTS,
            config.dims.NCHANS,
            config.dims.NTIME,
            config.dims.NPOLS,
            config.blockSize);

    BL_CHECK_THROW(this->InitInput(this->input.buf, getInputSize()));
    BL_CHECK_THROW(this->InitInput(this->input.phasors, getPhasorsSize()));
    BL_CHECK_THROW(this->InitOutput(this->output.buf, getOutputSize()));
}

template class MeerKAT<CF32, CF32>;

}  // namespace Blade::Modules::Beamformer
