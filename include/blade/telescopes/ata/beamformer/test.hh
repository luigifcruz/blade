#ifndef BLADE_TELESCOPES_ATA_BEAMFORMER_TEST_H
#define BLADE_TELESCOPES_ATA_BEAMFORMER_TEST_H

#include "blade/telescopes/generic/beamformer/test.hh"

namespace Blade::Telescope::ATA::Beamformer {

class BLADE_API Test : public Generic::Beamformer::Test {
public:
    Test();
    ~Test();

    Result beamform();

    std::span<const std::complex<int8_t>> getInputData();
    std::span<const std::complex<float>> getPhasorsData();
    std::span<const std::complex<float>> getOutputData();

private:
    py::scoped_interpreter guard{}; // WARNING: Interpreter should be destructed last!
    py::object lib;
};

} // namespace Blade::Telescope::ATA::Beamformer

#endif
