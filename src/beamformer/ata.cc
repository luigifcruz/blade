#include "blade/beamformer/ata.hh"

namespace Blade::Beamformer {

ATA::ATA(const Config & config) : Generic(config) {
    if (config.NBEAMS > config.blockSize) {
        BL_FATAL("The block size ({}) is smaller than the number of beams ({}).",
                config.blockSize, config.NBEAMS);
        throw Result::ERROR;
    }

    block = dim3(config.blockSize);
    grid = dim3(config.NCHANS, config.NTIME/config.blockSize);

    kernel = Template("ATA").instantiate(
        config.NBEAMS,
        config.NANTS,
        config.NCHANS,
        config.NTIME,
        config.NPOLS,
        config.blockSize
    );
}

} // namespace Blade::Beamformer
