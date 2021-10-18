#include "blade/channelizer/test.hh"

namespace Blade::Channelizer::Test {

GenericPython::GenericPython(const Channelizer::Generic::Config & config) {
    BL_DEBUG("Initilizating class.");
    lib = py::module::import("blade.instruments.channelizer.test").attr("Generic")(
            config.NANTS, config.NCHANS, config.NTIME, config.NPOLS, config.fftSize);
}
Result GenericPython::process() {
    BL_CATCH(lib.attr("process")(), [&]{
        BL_FATAL("Failed to execute Python function: {}", e.what());
        return Result::PYTHON_ERROR;
    });
    return Result::SUCCESS;
}

std::span<const std::complex<int8_t>> GenericPython::getInputData() {
    return getVector<int16_t, const std::complex<int8_t>>(lib.attr("getInputData"));
}

std::span<const std::complex<int8_t>> GenericPython::getOutputData() {
    return getVector<int16_t, const std::complex<int8_t>>(lib.attr("getOutputData"));
}

} // namespace Blade::Channelizer::Test
