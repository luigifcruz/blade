#ifndef BLADE_UTILS_CHECKER_HH
#define BLADE_UTILS_CHECKER_HH

#include "blade/base.hh"
#include "blade/memory/base.hh"

namespace Blade {

class BLADE_API Checker {
 public:
    template<typename IT, typename OT>
    static std::size_t run(const Vector<Device::CPU, IT>& a,
                           const Vector<Device::CPU, OT>& b);

    template<typename IT, typename OT>
    static std::size_t run(const Vector<Device::CPU, std::complex<IT>>& a,
                           const Vector<Device::CPU, std::complex<OT>>& b);

 private:
    template<typename IT, typename OT>
    static std::size_t run(IT a, OT b, std::size_t size, std::size_t scale = 1);
};

}  // namespace Blade

#endif
