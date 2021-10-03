#include "bl-beamformer/ata/beamformer/test.hh"

namespace BL::ATA::Beamformer {

Test::Test() {
    BL_DEBUG("Initilizating class.");
    lib = py::module::import("bl.ata.beamformer").attr("Test")();
}

Test::~Test() {
    BL_DEBUG("Destroying class.");
}

Result Test::beamform() {
    BL_DEBUG("Beamforming.");
    try {
        lib.attr("beamform")();
    } catch(...) {
        return Result::ERROR;
    }
    return Result::SUCCESS;
}

std::span<const std::complex<int8_t>> Test::getInputData() {
    return __convert<int16_t, const std::complex<int8_t>>(lib.attr("getInputData"));
}

std::span<const std::complex<float>> Test::getPhasorsData() {
    return __convert<std::complex<float>, const std::complex<float>>(lib.attr("getPhasorsData"));
}

std::span<const std::complex<float>> Test::getOutputData() {
    return __convert<std::complex<float>, const std::complex<float>>(lib.attr("getOutputData"));
}

template<typename IT, typename OT>
std::span<OT> Test::__convert(const py::object & input) {
    auto arr = input().cast<py::array_t<IT>>();
    auto buf = reinterpret_cast<OT*>(arr.data());
    auto len = static_cast<std::size_t>(arr.size());
    return std::span{buf, len};
}

} // namespace BL::ATA::Beamformer
