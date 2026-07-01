#ifndef BLADE_STACKER_MODULE_HH
#define BLADE_STACKER_MODULE_HH

#include <jetstream/module.hh>

namespace Jetstream::Modules {

struct Stacker : public Module::Config {
    U64 axis = 0;
    U64 ratio = 1;
    U64 copySizeThreshold = 512;
    U64 blockSize = 512;

    JST_MODULE_TYPE(stacker);
    JST_MODULE_PARAMS(axis, ratio, copySizeThreshold, blockSize);
};

}  // namespace Jetstream::Modules

#endif  // BLADE_STACKER_MODULE_HH
