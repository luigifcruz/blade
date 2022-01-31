#include "blade/modules/beamformer/ata.hh"

namespace Blade::Modules::Beamformer {

template<typename IT, typename OT>
ATA<IT, OT>::ATA(const typename Generic<IT, OT>::Config& config,
                 const typename Generic<IT, OT>::Input& input)
        : Generic<IT, OT>(config, input) {
    if (config.dims.NBEAMS > config.blockSize) {
        BL_FATAL("The block size ({}) is smaller than the number "
                "of beams ({}).", config.blockSize, config.dims.NBEAMS);
        BL_CHECK_THROW(Result::ERROR);
    }

    this->block = dim3(config.blockSize);
    this->grid = dim3(config.dims.NCHANS, config.dims.NTIME/config.blockSize);

    this->kernel = Template("ATA")
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

template class BLADE_API ATA<CF32, CF32>;

}  // namespace Blade::Modules::Beamformer
