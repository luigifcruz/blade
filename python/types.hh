#include "base.hh"

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

inline void init_types_xyz(const py::module& m) {
    py::class_<XYZ>(m, "XYZ", py::dynamic_attr())
        .def(py::init<F64, F64, F64>(), py::arg("X"),
                                        py::arg("Y"),
                                        py::arg("Z"))
        .def_readwrite("X", &XYZ::X)
        .def_readwrite("Y", &XYZ::Y)
        .def_readwrite("Z", &XYZ::Z);
}

inline void init_types_uvw(const py::module& m) {
    py::class_<UVW>(m, "UVW", py::dynamic_attr())
        .def(py::init<F64, F64, F64>(), py::arg("U"),
                                        py::arg("V"),
                                        py::arg("W"))
        .def_readwrite("U", &UVW::U)
        .def_readwrite("V", &UVW::V)
        .def_readwrite("W", &UVW::W);
}

inline void init_types_lla(const py::module& m) {
    py::class_<LLA>(m, "LLA", py::dynamic_attr())
        .def(py::init<F64, F64, F64>(), py::arg("LON"),
                                        py::arg("LAT"),
                                        py::arg("ALT"))
        .def_readwrite("LON", &LLA::LON)
        .def_readwrite("LAT", &LLA::LAT)
        .def_readwrite("ALT", &LLA::ALT);
}

inline void init_types_ra_dec(const py::module& m) {
    py::class_<RA_DEC>(m, "RA_DEC", py::dynamic_attr())
        .def(py::init<F64, F64>(), py::arg("RA"), py::arg("DEC"))
        .def_readwrite("RA", &RA_DEC::RA)
        .def_readwrite("DEC", &RA_DEC::DEC);
}

inline void init_types_ha_dec(const py::module& m) {
    py::class_<HA_DEC>(m, "HA_DEC", py::dynamic_attr())
        .def(py::init<F64, F64>(), py::arg("HA"), py::arg("DEC"))
        .def_readwrite("HA", &HA_DEC::HA)
        .def_readwrite("DEC", &HA_DEC::DEC);
}

inline void init_types(const py::module& m) {
    init_types_result(m);
    init_types_xyz(m);
    init_types_uvw(m);
    init_types_lla(m);
    init_types_ra_dec(m);
    init_types_ha_dec(m);
}
