#include <pybind11/pybind11.h>

#include "blade/base.hh"

#include "memory.hh"
#include "module.hh"

using namespace Blade;

PYBIND11_MODULE(blade, m) {
    m.def("BL_CHECK_THROW", [](const Result& result) {
        BL_CHECK_THROW(result);
    });

    m.def("BL_CUDA_CHECK_THROW", [](const cudaError_t& result) {
        BL_CUDA_CHECK_THROW(result, [&]{
            BL_FATAL("CUDA error: {}", err);
        });
    });

    py::enum_<Result>(m, "Result")
        .value("SUCCESS", Result::SUCCESS)
        .value("ERROR", Result::ERROR)
        .value("CUDA_ERROR", Result::CUDA_ERROR)
        .value("ASSERTION_ERROR", Result::ASSERTION_ERROR);

    py::enum_<Device>(m, "Device")
        .value("CPU", Device::CPU)
        .value("CUDA", Device::CUDA);

    init_memory(m);
    init_modules(m);
}
