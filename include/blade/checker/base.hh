#ifndef BLADE_CHECKER_GENERIC_H
#define BLADE_CHECKER_GENERIC_H

#include "blade/base.hh"
#include "blade/kernel.hh"

namespace Blade {

class BLADE_API Checker {
 public:
    template<typename IT, typename OT>
    unsigned long long int run(const std::span<IT>& a,
                               const std::span<OT>& b);

    template<typename IT, typename OT>
    unsigned long long int run(const std::span<std::complex<IT>>& a,
                               const std::span<std::complex<OT>>& b);

 private:
    unsigned long long int* counter;

    template<typename IT, typename OT>
    unsigned long long int run(IT a, OT b, std::size_t size, std::size_t scale = 1);
};

}  // namespace Blade

#endif  // BLADE_INCLUDE_BLADE_CHECKER_BASE_HH_
