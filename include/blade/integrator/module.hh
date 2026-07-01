#ifndef BLADE_INTEGRATOR_MODULE_HH
#define BLADE_INTEGRATOR_MODULE_HH

#include <jetstream/module.hh>

namespace Jetstream::Modules {

struct Integrator : public Module::Config {
    U64 size = 1;  // Number of indices to integrate within one block.
    U64 rate = 1;  // Number of blocks to integrate together.
    U64 axis = 2;  // The block axis to integrate on, defaulting to T
    U64 blockSize = 512;

    JST_MODULE_TYPE(integrator);
    JST_MODULE_PARAMS(size, rate, axis, blockSize);
};

}  // namespace Jetstream::Modules

#endif  // BLADE_INTEGRATOR_MODULE_HH
