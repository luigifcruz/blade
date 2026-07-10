#ifndef BLADE_CORRELATOR_MODULE_HH
#define BLADE_CORRELATOR_MODULE_HH

#include <string>

#include <jetstream/module.hh>

namespace Jetstream::Modules {

struct Correlator : public Module::Config {
    U64 integrationRate = 1;
    U64 conjugateAntennaIndex = 1;
    std::string calculationMode = "double_precision_fp";

    JST_MODULE_TYPE(correlator);
    JST_MODULE_PARAMS(integrationRate, conjugateAntennaIndex, calculationMode);
};

}  // namespace Jetstream::Modules

#endif  // BLADE_CORRELATOR_MODULE_HH
