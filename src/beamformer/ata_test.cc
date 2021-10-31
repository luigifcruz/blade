#include "blade/beamformer/ata_test.hh"

namespace Blade::Beamformer {

ATA::Test::Test(const Beamformer::Generic::Config& config) :
    GenericPython("ATA", config.dims) {}

}  // namespace Blade::Beamformer
