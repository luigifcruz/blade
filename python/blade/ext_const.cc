#include <nanobind/nanobind.h>

#include "blade/base.hh"

namespace nb = nanobind;
using namespace nb::literals;
using namespace Blade;

NB_MODULE(_const_impl, m) {
    nb::enum_<Result>(m, "result")
        .value("success", Result::SUCCESS)
        .value("error", Result::ERROR)
        .value("cuda_error", Result::CUDA_ERROR)
        .value("assertion_error", Result::ASSERTION_ERROR)
        .value("runner_queue_full", Result::RUNNER_QUEUE_FULL)
        .value("runner_queue_empty", Result::RUNNER_QUEUE_EMPTY)
        .value("runner_queue_none_available", Result::RUNNER_QUEUE_NONE_AVAILABLE)
        .value("pipeline_exhausted", Result::PIPELINE_EXHAUSTED)
        .export_values();

    nb::class_<Stream>(m, "stream")
        .def("__init__", [](Stream* stream) {
            static auto _stream = Stream{nullptr};
            stream = &_stream;
        })
        .def("__repr__", [](Stream& obj){
            return fmt::format("Stream()");
        });

    nb::enum_<POL>(m, "pol")
        .value("x", POL::X)
        .value("y", POL::Y)
        .value("l", POL::L)
        .value("r", POL::R)
        .value("xy", POL::XY)
        .value("lr", POL::LR)
        .export_values();

    nb::class_<XYZ>(m, "xyz", nb::dynamic_attr())
        .def(nb::init<F64, F64, F64>(), "x"_a, "y"_a, "z"_a)
        .def("__init__", [](XYZ* t, const std::tuple<F64, F64, F64>& elements) {
            const auto& [a, b, c] = elements;
            new (t) XYZ(a, b, c);
        })
        .def_rw("x", &XYZ::X)
        .def_rw("y", &XYZ::Y)
        .def_rw("z", &XYZ::Z)
        .def("__repr__", [](XYZ& obj){
            return fmt::format("XYZ(x={}, y={}, z={})", obj.X, obj.Y, obj.Z);
        });
    nb::implicitly_convertible<std::tuple<F64, F64, F64>, XYZ>();

    nb::class_<UVW>(m, "uvm", nb::dynamic_attr())
        .def(nb::init<F64, F64, F64>(), "u"_a, "v"_a, "w"_a)
        .def("__init__", [](UVW* t, const std::tuple<F64, F64, F64>& elements) {
            const auto& [a, b, c] = elements;
            new (t) UVW(a, b, c);
        })
        .def_rw("u", &UVW::U)
        .def_rw("v", &UVW::V)
        .def_rw("w", &UVW::W)
        .def("__repr__", [](UVW& obj){
            return fmt::format("UVW(u={}, v={}, w={})", obj.U, obj.V, obj.W);
        });
    nb::implicitly_convertible<std::tuple<F64, F64, F64>, UVW>();

    nb::class_<LLA>(m, "lla", nb::dynamic_attr())
        .def(nb::init<F64, F64, F64>(), "lon"_a, "lat"_a, "alt"_a)
        .def("__init__", [](LLA* t, const std::tuple<F64, F64, F64>& elements) {
            const auto& [a, b, c] = elements;
            new (t) LLA(a, b, c);
        })
        .def_rw("lon", &LLA::LON)
        .def_rw("lat", &LLA::LAT)
        .def_rw("alt", &LLA::ALT)
        .def("__repr__", [](LLA& obj){
            return fmt::format("LLA(lon={}, lat={}, alt={})", obj.LON, obj.LAT, obj.ALT);
        });
    nb::implicitly_convertible<std::tuple<F64, F64, F64>, LLA>();

    nb::class_<RA_DEC>(m, "ra_dec", nb::dynamic_attr())
        .def(nb::init<F64, F64>(), "ra"_a, "dec"_a)
        .def("__init__", [](RA_DEC* t, const std::tuple<F64, F64>& elements) {
            const auto& [a, b] = elements;
            new (t) RA_DEC(a, b);
        })
        .def_rw("ra", &RA_DEC::RA)
        .def_rw("dec", &RA_DEC::DEC)
        .def("__repr__", [](RA_DEC& obj){
            return fmt::format("RA_DEC(ra={}, dec={})", obj.RA, obj.DEC);
        });
    nb::implicitly_convertible<std::tuple<F64, F64>, RA_DEC>();

    nb::class_<HA_DEC>(m, "ha_dec", nb::dynamic_attr())
        .def(nb::init<F64, F64>(), "ha"_a, "dec"_a)
        .def("__init__", [](HA_DEC* t, const std::tuple<F64, F64>& elements) {
            const auto& [a, b] = elements;
            new (t) HA_DEC(a, b);
        })
        .def_rw("ha", &HA_DEC::HA)
        .def_rw("dec", &HA_DEC::DEC)
        .def("__repr__", [](HA_DEC& obj){
            return fmt::format("HA_DEC(ha={}, dec={})", obj.HA, obj.DEC);
        });
    nb::implicitly_convertible<std::tuple<F64, F64>, HA_DEC>();
}
