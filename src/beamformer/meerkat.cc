#include "blade/beamformer/meerkat.hh"

namespace Blade::Beamformer {

MeerKAT::MeerKAT(const Config& config) : Generic(config) {
    block = dim3(config.blockSize);
    grid = dim3(config.dims.NCHANS, config.dims.NTIME/config.blockSize);

    kernel = Template("MeerKAT").instantiate(
        config.dims.NBEAMS,
        config.dims.NANTS,
        config.dims.NCHANS,
        config.dims.NTIME,
        config.dims.NPOLS,
        config.blockSize);
}

}  // namespace Blade::Beamformer
