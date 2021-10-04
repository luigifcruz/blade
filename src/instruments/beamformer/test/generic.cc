#include "blade/instruments/beamformer/test/generic.hh"

namespace Blade::Instrument::Beamformer::Test {

Generic::Generic(const std::string & module_str) {
    BL_DEBUG("Initilizating class.");
    lib = py::module::import(module_str.c_str()).attr("Test")();
}

Result Generic::beamform() {
    try {
        lib.attr("beamform")();
    } catch(...) {
        BL_FATAL("Failed to execute Python function.");
        return Result::ERROR;
    }
    return Result::SUCCESS;
}

std::span<const std::complex<int8_t>> Generic::getInputData() {
    return __convert<int16_t, const std::complex<int8_t>>(lib.attr("getInputData"));
}

std::span<const std::complex<float>> Generic::getPhasorsData() {
    return __convert<std::complex<float>, const std::complex<float>>(lib.attr("getPhasorsData"));
}

std::span<const std::complex<float>> Generic::getOutputData() {
    return __convert<std::complex<float>, const std::complex<float>>(lib.attr("getOutputData"));
}

} // namespace Blade::Generic::Beamformer::Test
