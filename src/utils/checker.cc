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
std::size_t Checker::run(const Vector<Device::CPU, std::complex<IT>>& a,
                         const Vector<Device::CPU, std::complex<OT>>& b) {
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
std::size_t Checker::run(const Vector<Device::CPU, IT>& a,
                         const Vector<Device::CPU, OT>& b) {
    if (a.size() != b.size()) {
        BL_FATAL("Size mismatch between checker inputs.");
        return -1;
    }

    return Checker::run(a.data(), b.data(), a.size());
}

template std::size_t Checker::run(const Vector<Device::CPU, CF32>&,
                                  const Vector<Device::CPU, CF32>&);

template std::size_t Checker::run(const Vector<Device::CPU, CI8>&,
                                  const Vector<Device::CPU, CI8>&);

template std::size_t Checker::run(const Vector<Device::CPU, CF16>&,
                                  const Vector<Device::CPU, CF16>&);

template std::size_t Checker::run(const Vector<Device::CPU, F32>&,
                                  const Vector<Device::CPU, F32>&);

template std::size_t Checker::run(const Vector<Device::CPU, I8>&,
                                  const Vector<Device::CPU, I8>&);

template std::size_t Checker::run(const Vector<Device::CPU, F16>&,
                                  const Vector<Device::CPU, F16>&);

}  // namespace Blade::Modules
