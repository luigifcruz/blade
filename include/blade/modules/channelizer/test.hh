#ifndef BLADE_MODULES_CHANNELIZER_TEST_GENERIC_H
#define BLADE_MODULES_CHANNELIZER_TEST_GENERIC_H

#include "blade/base.hh"
#include "blade/python.hh"
#include "blade/modules/channelizer/base.hh"

namespace Blade::Modules {

class BLADE_API Channelizer::Test : protected python {
 public:
    explicit Test(const Channelizer::Config& config);
    ~Test() = default;

    Result process();

    std::span<CF32> getInputData();
    std::span<CF32> getOutputData();
};

}  // namespace Blade::Modules

#endif
