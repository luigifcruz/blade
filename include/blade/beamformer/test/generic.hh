#ifndef BLADE_BEAMFORMER_TEST_GENERIC_H
#define BLADE_BEAMFORMER_TEST_GENERIC_H

#include "blade/base.hh"
#include "blade/python.hh"

namespace Blade::Beamformer::Test {

class BLADE_API Generic : protected Python {
public:
    Generic(const std::string & telescope);
    ~Generic() = default;

    Result beamform();

    std::span<const std::complex<int8_t>> getInputData();
    std::span<const std::complex<float>> getPhasorsData();
    std::span<const std::complex<float>> getOutputData();
};

} // namespace Blade::Beamformer::Test

#endif
