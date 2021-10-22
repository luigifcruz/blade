#include "blade/beamformer/test/generic.hh"

namespace Blade::Beamformer::Test {

GenericPython::GenericPython(const std::string & telescope,
                             const ArrayDims & dims) {
    BL_DEBUG("Initilizating class.");
    lib = py::module::import("blade.instruments.beamformer.test")
        .attr(telescope.c_str())(dims.NBEAMS, dims.NANTS, dims.NCHANS,
                                 dims.NTIME, dims.NPOLS);
}

Result GenericPython::process() {
    BL_CATCH(lib.attr("process")(), [&]{
        BL_FATAL("Failed to execute Python function: {}", e.what());
        return Result::PYTHON_ERROR;
    });
    return Result::SUCCESS;
}

std::span<const std::complex<float>> GenericPython::getInputData() {
    return getVector<std::complex<float>, const std::complex<float>>
        (lib.attr("getInputData"));
}

std::span<const std::complex<float>> GenericPython::getPhasorsData() {
    return getVector<std::complex<float>, const std::complex<float>>
        (lib.attr("getPhasorsData"));
}

std::span<const std::complex<float>> GenericPython::getOutputData() {
    return getVector<std::complex<float>, const std::complex<float>>
        (lib.attr("getOutputData"));
}

} // namespace Blade::Beamformer::Test
