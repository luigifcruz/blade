#ifndef BLADE_BEAMFORMER_TEST_GENERIC_H
#define BLADE_BEAMFORMER_TEST_GENERIC_H

#include <string>

#include "blade/base.hh"
#include "blade/python.hh"
#include "blade/beamformer/generic.hh"

namespace Blade::Beamformer {

class BLADE_API Generic::Test {
 public:
    virtual ~Test() = default;

    virtual Result process() = 0;

    virtual std::span<CF32> getInputData() = 0;
    virtual std::span<CF32> getPhasorsData() = 0;
    virtual std::span<CF32> getOutputData() = 0;
};

class BLADE_API GenericPython : public Generic::Test, protected Python {
 public:
    explicit GenericPython(const std::string& telescope, const ArrayDims& dims);
    ~GenericPython() = default;

    Result process();

    std::span<CF32> getInputData();
    std::span<CF32> getPhasorsData();
    std::span<CF32> getOutputData();
};

}  // namespace Blade::Beamformer

#endif  // BLADE_INCLUDE_BLADE_BEAMFORMER_GENERIC_TEST_HH_
