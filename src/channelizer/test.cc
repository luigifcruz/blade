#include "blade/channelizer/test.hh"

namespace Blade::Channelizer::Test {

Generic::Generic(const Channelizer::Generic::Config & config) {
    BL_DEBUG("Initilizating class.");
    lib = py::module::import("blade.instruments.channelizer.test").attr("Generic")(
            config.NANTS, config.NCHANS, config.NTIME, config.NPOLS, config.fftSize);
}
Result Generic::process() {
    BL_CATCH(lib.attr("process")(), [&]{
        BL_FATAL("Failed to execute Python function: {}", e.what());
        return Result::PYTHON_ERROR;
    });
    return Result::SUCCESS;
}

std::span<const std::complex<float>> Generic::getInputData() {
    return getVector<std::complex<float>, const std::complex<float>>
        (lib.attr("getInputData"));
}

std::span<const std::complex<float>> Generic::getOutputData() {
    return getVector<std::complex<float>, const std::complex<float>>
        (lib.attr("getOutputData"));
}

} // namespace Blade::Channelizer::Test
