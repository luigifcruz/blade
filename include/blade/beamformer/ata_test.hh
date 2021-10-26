#ifndef BLADE_BEAMFORMER_TEST_ATA_H
#define BLADE_BEAMFORMER_TEST_ATA_H

#include "blade/beamformer/generic_test.hh"
#include "blade/beamformer/ata.hh"

namespace Blade::Beamformer {

class BLADE_API ATA::Test : public GenericPython {
public:
    explicit Test(const Beamformer::Generic::Config& config);
    ~Test() = default;
};

} // namespace Blade::Beamformer

#endif
