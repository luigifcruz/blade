#ifndef BLADE_CHANNELIZER_TEST_GENERIC_H
#define BLADE_CHANNELIZER_TEST_GENERIC_H

#include "blade/base.hh"
#include "blade/python.hh"
#include "blade/channelizer/base.hh"

namespace Blade::Channelizer::Test {

class BLADE_API Generic {
public:
    virtual ~Generic() = default;

    virtual Result process() = 0;

    virtual std::span<const std::complex<int8_t>> getInputData() = 0;
    virtual std::span<const std::complex<int8_t>> getOutputData() = 0;
};

class BLADE_API GenericPython : public Generic, protected Python {
public:
    GenericPython(const Channelizer::Generic::Config & config);
    ~GenericPython() = default;

    Result process();

    std::span<const std::complex<int8_t>> getInputData();
    std::span<const std::complex<int8_t>> getOutputData();
};

} // namespace Blade::Channelizer::Test

#endif
