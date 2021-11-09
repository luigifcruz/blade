#include "blade/modules/beamformer/ata.hh"

namespace Blade::Modules::Beamformer {

ATA::ATA(const Config& config) : Generic(config) {
    if (config.dims.NBEAMS > config.blockSize) {
        BL_FATAL("The block size ({}) is smaller than the number "
                "of beams ({}).", config.blockSize, config.dims.NBEAMS);
        throw Result::ERROR;
    }

    block = dim3(config.blockSize);
    grid = dim3(config.dims.NCHANS, config.dims.NTIME/config.blockSize);

    kernel = Template("ATA").instantiate(
        config.dims.NBEAMS,
        config.dims.NANTS,
        config.dims.NCHANS,
        config.dims.NTIME,
        config.dims.NPOLS,
        config.blockSize);
}

}  // namespace Blade::Modules::Beamformer
