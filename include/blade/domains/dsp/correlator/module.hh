#ifndef BLADE_DOMAINS_DSP_CORRELATOR_MODULE_HH
#define BLADE_DOMAINS_DSP_CORRELATOR_MODULE_HH

#include <string>

#include <jetstream/module.hh>

namespace Jetstream::Modules {

struct Correlator : public Module::Config {
    U64 integrationRate = 1;
    U64 conjugateAntennaIndex = 1;
    bool useSharedMemory = false;
    std::string calculationMode = "double_precision_fp";
    U64 blockSize = 32;

    JST_MODULE_TYPE(correlator);
    JST_MODULE_PARAMS(integrationRate, conjugateAntennaIndex, useSharedMemory,
                      calculationMode, blockSize);
};

}  // namespace Jetstream::Modules

#endif  // BLADE_DOMAINS_DSP_CORRELATOR_MODULE_HH
