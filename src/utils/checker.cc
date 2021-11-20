#include "blade/utils/checker.hh"

namespace Blade {

template<typename IT, typename OT>
std::size_t Checker::run(IT a, OT b, std::size_t size, std::size_t scale) {
    std::size_t counter = 0;
    for (std::size_t i = 0; i < (size * scale); i++) {
        if (abs(static_cast<double>(a[i]) - static_cast<double>(b[i])) > 0.1) {
            counter += 1;
        }
    }
    return counter / scale;
}

template<typename IT, typename OT>
std::size_t Checker::run(const Memory::HostVector<std::complex<IT>>& a,
                         const Memory::HostVector<std::complex<OT>>& b) {
    if (a.size() != b.size()) {
        BL_FATAL("Size mismatch between checker inputs.");
        return -1;
    }

    return Checker::run(
        reinterpret_cast<const IT*>(a.data()),
        reinterpret_cast<const OT*>(b.data()),
        a.size(), 2);
}

template<typename IT, typename OT>
std::size_t Checker::run(const Memory::HostVector<IT>& a,
                         const Memory::HostVector<OT>& b) {
    if (a.size() != b.size()) {
        BL_FATAL("Size mismatch between checker inputs.");
        return -1;
    }

    return Checker::run(a.data(), b.data(), a.size());
}

template std::size_t Checker::run(const Memory::HostVector<CF32>&,
                                  const Memory::HostVector<CF32>&);

template std::size_t Checker::run(const Memory::HostVector<CI8>&,
                                  const Memory::HostVector<CI8>&);

template std::size_t Checker::run(const Memory::HostVector<CF16>&,
                                  const Memory::HostVector<CF16>&);

template std::size_t Checker::run(const Memory::HostVector<F32>&,
                                  const Memory::HostVector<F32>&);

template std::size_t Checker::run(const Memory::HostVector<I8>&,
                                  const Memory::HostVector<I8>&);

template std::size_t Checker::run(const Memory::HostVector<F16>&,
                                  const Memory::HostVector<F16>&);

}  // namespace Blade::Modules
