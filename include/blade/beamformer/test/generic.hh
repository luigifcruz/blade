#ifndef BLADE_BEAMFORMER_TEST_GENERIC_H
#define BLADE_BEAMFORMER_TEST_GENERIC_H

#include "blade/base.hh"
#include "blade/python.hh"

namespace Blade::Beamformer::Test {

class BLADE_API Generic {
public:
    virtual ~Generic() = default;

    virtual Result process() = 0;

    virtual std::span<const std::complex<float>> getInputData() = 0;
    virtual std::span<const std::complex<float>> getPhasorsData() = 0;
    virtual std::span<const std::complex<float>> getOutputData() = 0;
};

class BLADE_API GenericPython : public Generic, protected Python {
public:
    GenericPython(const std::string & telescope, const ArrayDims & dims);
    ~GenericPython() = default;

    Result process();

    std::span<const std::complex<float>> getInputData();
    std::span<const std::complex<float>> getPhasorsData();
    std::span<const std::complex<float>> getOutputData();
};

} // namespace Blade::Beamformer::Test

#endif
