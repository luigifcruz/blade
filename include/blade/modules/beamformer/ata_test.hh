#ifndef BLADE_MODULES_BEAMFORMER_TEST_ATA_H
#define BLADE_MODULES_BEAMFORMER_TEST_ATA_H

#include "blade/modules/beamformer/generic_test.hh"
#include "blade/modules/beamformer/ata.hh"

namespace Blade::Modules::Beamformer {

class BLADE_API ATA::Test : public GenericPython {
 public:
    explicit Test(const Beamformer::Generic::Config& config);
    ~Test() = default;
};

}  // namespace Blade::Modules::Beamformer

#endif
