#ifndef BLADE_UTILS_CHECKER_HH
#define BLADE_UTILS_CHECKER_HH

#include "blade/base.hh"
#include "blade/memory.hh"

namespace Blade {

class BLADE_API Checker {
 public:
    template<typename IT, typename OT>
    static std::size_t run(const Memory::HostVector<IT>& a,
                           const Memory::HostVector<OT>& b);

    template<typename IT, typename OT>
    static std::size_t run(const Memory::HostVector<std::complex<IT>>& a,
                           const Memory::HostVector<std::complex<OT>>& b);

 private:
    template<typename IT, typename OT>
    static std::size_t run(IT a, OT b, std::size_t size, std::size_t scale = 1);
};

}  // namespace Blade

#endif
