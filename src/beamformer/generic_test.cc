#include "blade/beamformer/generic_test.hh"

namespace Blade::Beamformer {

GenericPython::GenericPython(const std::string& telescope, const ArrayDims& dims) {
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

std::span<std::complex<F32>> GenericPython::getInputData() {
    return getVector<std::complex<F32>, std::complex<F32>>
        (lib.attr("getInputData"));
}

std::span<std::complex<F32>> GenericPython::getPhasorsData() {
    return getVector<std::complex<F32>, std::complex<F32>>
        (lib.attr("getPhasorsData"));
}

std::span<std::complex<F32>> GenericPython::getOutputData() {
    return getVector<std::complex<F32>, std::complex<F32>>
        (lib.attr("getOutputData"));
}

}  // namespace Blade::Beamformer
