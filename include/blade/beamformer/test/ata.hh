#ifndef BLADE_BEAMFORMER_TEST_ATA_H
#define BLADE_BEAMFORMER_TEST_ATA_H

#include "blade/beamformer/test/generic.hh"
#include "blade/beamformer/ata.hh"

namespace Blade::Beamformer::Test {

class BLADE_API ATA : public GenericPython {
public:
    ATA(const Beamformer::Generic::Config & config);
    ~ATA() = default;
};

} // namespace Blade::Beamformer::Test

#endif
