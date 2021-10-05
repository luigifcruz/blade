#include "blade/beamformer/ata.hh"

namespace Blade::Beamformer {

ATA::ATA(const Config & config) : Generic(config) {
    if (config.NBEAMS > config.TBLOCK) {
        BL_FATAL("TBLOCK is smaller than NBEAMS.");
        throw Result::ERROR;
    }

    block = dim3(config.TBLOCK);
    grid = dim3(config.NCHANS, config.NTIME/config.TBLOCK);

    kernel = Template("ATA").instantiate(
        config.NBEAMS,
        config.NANTS,
        config.NCHANS,
        config.NTIME,
        config.NPOLS,
        config.TBLOCK
    );
}

} // namespace Blade::Beamformer
