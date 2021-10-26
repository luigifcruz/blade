#include "blade/beamformer/meerkat.hh"

namespace Blade::Beamformer {

MeerKAT::MeerKAT(const Config& config) : Generic(config) {
    block = dim3(config.blockSize);
    grid = dim3(config.NCHANS, config.NTIME/config.blockSize);

    kernel = Template("MeerKAT").instantiate(
        config.NBEAMS,
        config.NANTS,
        config.NCHANS,
        config.NTIME,
        config.NPOLS,
        config.blockSize
    );
}

} // namespace Blade::Beamformer
