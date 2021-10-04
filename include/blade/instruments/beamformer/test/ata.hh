#ifndef BLADE_INSTRUMENTS_BEAMFORMER_TEST_ATA_H
#define BLADE_INSTRUMENTS_BEAMFORMER_TEST_ATA_H

#include "blade/instruments/beamformer/test/generic.hh"

namespace Blade::Instrument::Beamformer::Test {

class BLADE_API ATA : public Generic {
public:
    ATA();
    ~ATA() = default;
};

} // namespace Blade::Instrument::Beamformer::Test

#endif
