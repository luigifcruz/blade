#ifndef BLADE_BEAMFORMER_TEST_GENERIC_H
#define BLADE_BEAMFORMER_TEST_GENERIC_H

#include "blade/base.hh"
#include "blade/python.hh"
#include "blade/beamformer/generic.hh"

namespace Blade::Beamformer {

class BLADE_API Generic::Test {
public:
    virtual ~Test() = default;

    virtual Result process() = 0;

    virtual std::span<std::complex<float>> getInputData() = 0;
    virtual std::span<std::complex<float>> getPhasorsData() = 0;
    virtual std::span<std::complex<float>> getOutputData() = 0;
};

class BLADE_API GenericPython : public Generic::Test, protected Python {
public:
    explicit GenericPython(const std::string& telescope, const ArrayDims& dims);
    ~GenericPython() = default;

    Result process();

    std::span<std::complex<float>> getInputData();
    std::span<std::complex<float>> getPhasorsData();
    std::span<std::complex<float>> getOutputData();
};

} // namespace Blade::Beamformer

#endif
