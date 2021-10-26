#include "blade/channelizer/test.hh"

namespace Blade {

Channelizer::Test::Test(const Channelizer::Config& config) {
    BL_DEBUG("Initilizating class.");
    lib = py::module::import("blade.instruments.channelizer.test").attr("Generic")(
            config.NANTS, config.NCHANS, config.NTIME, config.NPOLS, config.fftSize);
}

Result Channelizer::Test::process() {
    BL_CATCH(lib.attr("process")(), [&]{
        BL_FATAL("Failed to execute Python function: {}", e.what());
        return Result::PYTHON_ERROR;
    });
    return Result::SUCCESS;
}

std::span<std::complex<float>> Channelizer::Test::getInputData() {
    return getVector<std::complex<float>, std::complex<float>>
        (lib.attr("getInputData"));
}

std::span<std::complex<float>> Channelizer::Test::getOutputData() {
    return getVector<std::complex<float>, std::complex<float>>
        (lib.attr("getOutputData"));
}

} // namespace Blade
