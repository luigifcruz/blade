#include "blade/modules/checker/base.hh"

namespace Blade::Modules {

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
std::size_t Checker::run(const std::span<std::complex<IT>>& a,
                         const std::span<std::complex<OT>>& b) {
    if (a.size() != b.size()) {
        BL_FATAL("Size mismatch between checker inputs.");
        return -1;
    }

    return this->run(
        reinterpret_cast<const IT*>(a.data()),
        reinterpret_cast<const OT*>(b.data()),
        a.size(), 2);
}

template<typename IT, typename OT>
std::size_t Checker::run(const std::span<IT>& a,
                         const std::span<OT>& b) {
    if (a.size() != b.size()) {
        BL_FATAL("Size mismatch between checker inputs.");
        return -1;
    }

    return this->run(a.data(), b.data(), a.size());
}

template std::size_t Checker::run(const std::span<CF32>&,
                                  const std::span<CF32>&);

template std::size_t Checker::run(const std::span<CI8>&,
                                  const std::span<CI8>&);

template std::size_t Checker::run(const std::span<CF16>&,
                                  const std::span<CF16>&);

template std::size_t Checker::run(const std::span<F32>&,
                                  const std::span<F32>&);

template std::size_t Checker::run(const std::span<I8>&,
                                  const std::span<I8>&);

template std::size_t Checker::run(const std::span<F16>&,
                                  const std::span<F16>&);

}  // namespace Blade::Modules
