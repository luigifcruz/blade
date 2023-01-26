#ifndef BLADE_MODULES_CHANNELIZER_CALLBACK_HH
#define BLADE_MODULES_CHANNELIZER_CALLBACK_HH

#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>

#include "blade/logger.hh"
#include "blade/macros.hh"

namespace Blade {
namespace Internal {

class Callback {
 public:
    Callback(cufftHandle& plan);

 private:
    cufftCallbackLoadC h_loadCallbackPtr;
};

}  // namespace Internal
}  // namespace Blade

#endif
