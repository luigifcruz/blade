#include "blade/beamformer/test/ata.hh"

namespace Blade::Beamformer::Test {

ATA::ATA(const Beamformer::Generic::Config & config) :
    GenericPython("ATA", config) {}

} // namespace Blade::Beamformer::Test
