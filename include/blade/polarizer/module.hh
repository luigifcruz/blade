#ifndef BLADE_POLARIZER_MODULE_HH
#define BLADE_POLARIZER_MODULE_HH

#include <jetstream/module.hh>

namespace Jetstream::Modules {

struct Polarizer : public Module::Config {
    std::string inputPolarization = "xy";
    std::string outputPolarization = "lr";
    U64 blockSize = 512;

    JST_MODULE_TYPE(polarizer);
    JST_MODULE_PARAMS(inputPolarization, outputPolarization, blockSize);
};

}  // namespace Jetstream::Modules

#endif  // BLADE_POLARIZER_MODULE_HH
