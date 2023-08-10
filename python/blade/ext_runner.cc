#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/shared_ptr.h>

#include "blade/base.hh"
#include "blade/memory/custom.hh"

namespace nb = nanobind;
using namespace nb::literals;
using namespace Blade;

NB_MODULE(_runner_impl, m) {
    nb::class_<Runner>(m, "runner")
        .def(nb::init<const std::shared_ptr<Pipeline>&>(), "pipeline"_a)
        .def("enqueue", &Runner::enqueue, nb::rv_policy::reference)
        .def("dequeue", &Runner::dequeue, nb::rv_policy::reference)
        .def("__repr__", [](Runner& obj){
            return fmt::format("Runner()");
        });
}