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

inline void init_types_arraydims(const py::module& m) {
    py::class_<ArrayDims>(m, "ArrayDims")
        .def(py::init<const std::size_t&,
                      const std::size_t&,
                      const std::size_t&,
                      const std::size_t&,
                      const std::size_t&>(), py::arg("NBEAMS"),
                                             py::arg("NANTS"),
                                             py::arg("NCHANS"),
                                             py::arg("NTIME"),
                                             py::arg("NPOLS"))
        .def_property_readonly("NBEAMS", [](ArrayDims& obj){
            return obj.NBEAMS;
        })
        .def_property_readonly("NANTS", [](ArrayDims& obj){
            return obj.NANTS;
        })
        .def_property_readonly("NCHANS", [](ArrayDims& obj){
            return obj.NCHANS;
        })
        .def_property_readonly("NTIME", [](ArrayDims& obj){
            return obj.NTIME;
        })
        .def_property_readonly("NPOLS", [](ArrayDims& obj){
            return obj.NPOLS;
        })
        .def("size", [](ArrayDims& obj){
            return obj.getSize();
        });
}

inline void init_types(const py::module& m) {
    init_types_result(m);
    init_types_arraydims(m);
}
