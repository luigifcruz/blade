#include "blade/utils/checker.hh"

namespace Blade {

template<typename IT, typename OT>
U64 Checker::run(IT a, OT b, U64 size, U64 scale) {
    U64 counter = 0;
    for (U64 i = 0; i < (size * scale); i++) {
        if (abs(static_cast<double>(a[i]) - static_cast<double>(b[i])) > 0.1) {
            counter += 1;
        }
    }
    return counter / scale;
}

template<typename IT, typename OT>
U64 Checker::run(const ArrayTensor<Device::CPU, std::complex<IT>>& a,
                         const ArrayTensor<Device::CPU, std::complex<OT>>& b) {
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
U64 Checker::run(const ArrayTensor<Device::CPU, IT>& a,
                         const ArrayTensor<Device::CPU, OT>& b) {
    if (a.size() != b.size()) {
        BL_FATAL("Size mismatch between checker inputs.");
        return -1;
    }

    return Checker::run(a.data(), b.data(), a.size());
}

template U64 BLADE_API Checker::run(const ArrayTensor<Device::CPU, CF32>&,
                                    const ArrayTensor<Device::CPU, CF32>&);

template U64 BLADE_API Checker::run(const ArrayTensor<Device::CPU, CI8>&,
                                    const ArrayTensor<Device::CPU, CI8>&);

template U64 BLADE_API Checker::run(const ArrayTensor<Device::CPU, CF16>&,
                                    const ArrayTensor<Device::CPU, CF16>&);

template U64 BLADE_API Checker::run(const ArrayTensor<Device::CPU, F32>&,
                                    const ArrayTensor<Device::CPU, F32>&);

template U64 BLADE_API Checker::run(const ArrayTensor<Device::CPU, I8>&,
                                    const ArrayTensor<Device::CPU, I8>&);

template U64 BLADE_API Checker::run(const ArrayTensor<Device::CPU, F16>&,
                                    const ArrayTensor<Device::CPU, F16>&);

}  // namespace Blade::Modules
