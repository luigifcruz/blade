#ifndef BLADE_UTILS_CHECKER_HH
#define BLADE_UTILS_CHECKER_HH

#include "blade/base.hh"
#include "blade/memory/base.hh"

namespace Blade {

class BLADE_API Checker {
 public:
    template<typename IT, typename OT>
    static U64 run(const Vector<Device::CPU, IT>& a,
                           const Vector<Device::CPU, OT>& b);

    template<typename IT, typename OT>
    static U64 run(const Vector<Device::CPU, std::complex<IT>>& a,
                           const Vector<Device::CPU, std::complex<OT>>& b);

 private:
    template<typename IT, typename OT>
    static U64 run(IT a, OT b, U64 size, U64 scale = 1);
};

}  // namespace Blade

#endif
