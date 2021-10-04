#include "blade/instruments/beamformer/test/generic.hh"

namespace Blade::Instrument::Beamformer::Test {

Generic::Generic(const std::string & telescope) {
    BL_DEBUG("Initilizating class.");
    lib = py::module::import("blade.instruments.beamformer.test").attr(telescope.c_str())();
}

Result Generic::beamform() {
    BL_CATCH(lib.attr("beamform")(), [&]{
        BL_FATAL("Failed to execute Python function: {}", e.what());
        return Result::PYTHON_ERROR;
    });
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
