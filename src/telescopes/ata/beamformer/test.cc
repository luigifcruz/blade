#include "blade/telescopes/ata/beamformer/test.hh"

namespace Blade::Telescope::ATA::Beamformer {

Test::Test() {
    BL_DEBUG("Initilizating class.");
    lib = py::module::import("blade.ata.beamformer").attr("Test")();
}

Test::~Test() {
    BL_DEBUG("Destroying class.");
}

Result Test::beamform() {
    try {
        lib.attr("beamform")();
    } catch(...) {
        BL_FATAL("Failed to execute Python function.");
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

} // namespace Blade::Telescope::ATA::Beamformer
