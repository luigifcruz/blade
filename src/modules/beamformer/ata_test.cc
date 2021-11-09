#include "blade/modules/beamformer/ata_test.hh"

namespace Blade::Modules::Beamformer {

ATA::Test::Test(const Beamformer::Generic::Config& config) :
    GenericPython("ATA", config.dims) {}

}  // namespace Blade::Modules::Beamformer
