#include "blade/beamformer/meerkat.hh"

namespace Blade::Beamformer {

MeerKAT::MeerKAT(const Config & config) : Generic(config) {
    block = dim3(config.TBLOCK);
    grid = dim3(config.NCHANS, config.NTIME/config.TBLOCK);

    kernel = Template("MeerKAT").instantiate(
        config.NBEAMS,
        config.NANTS,
        config.NCHANS,
        config.NTIME,
        config.NPOLS,
        config.TBLOCK
    );
}

} // namespace Blade::Beamformer
