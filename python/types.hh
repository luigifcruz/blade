#include <pybind11/pybind11.h>

#include <blade/base.hh>

using namespace Blade;
namespace py = pybind11;

inline void init_types_result(const py::module& m) {
    py::enum_<Result>(m, "Result")
        .value("SUCCESS", Result::SUCCESS)
        .value("ERROR", Result::ERROR)
        .value("CUDA_ERROR", Result::CUDA_ERROR)
        .value("ASSERTION_ERROR", Result::ASSERTION_ERROR);
}

inline void init_types(const py::module& m) {
    init_types_result(m);
}
