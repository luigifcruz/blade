#ifndef BLADE_INSTRUMENTS_BEAMFORMER_TEST_GENERIC_H
#define BLADE_INSTRUMENTS_BEAMFORMER_TEST_GENERIC_H

#include "blade/instruments/base.hh"

namespace Blade::Instrument::Beamformer::Test {

class BLADE_API Generic : protected Instrument::Generic {
public:
    Generic(const std::string & telescope);
    ~Generic() = default;

    Result beamform();

    std::span<const std::complex<int8_t>> getInputData();
    std::span<const std::complex<float>> getPhasorsData();
    std::span<const std::complex<float>> getOutputData();

private:
    py::scoped_interpreter guard{}; // WARNING: Interpreter should be destructed last!
    py::object lib;
};

} // namespace Blade::Instrument::Beamformer::Test

#endif
