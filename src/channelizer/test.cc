#include "blade/channelizer/test.hh"

namespace Blade {

Channelizer::Test::Test(const Channelizer::Config& config) {
    BL_DEBUG("Initilizating class.");
    lib = py::module::import("blade.instruments.channelizer.test").attr("Generic")(
            config.dims.NANTS, config.dims.NCHANS, config.dims.NTIME,
            config.dims.NPOLS, config.fftSize);
}

Result Channelizer::Test::process() {
    BL_CATCH(lib.attr("process")(), [&]{
        BL_FATAL("Failed to execute Python function: {}", e.what());
        return Result::PYTHON_ERROR;
    });
    return Result::SUCCESS;
}

std::span<CF32> Channelizer::Test::getInputData() {
    return getVector<CF32, CF32>
        (lib.attr("getInputData"));
}

std::span<CF32> Channelizer::Test::getOutputData() {
    return getVector<CF32, CF32>
        (lib.attr("getOutputData"));
}

}  // namespace Blade
