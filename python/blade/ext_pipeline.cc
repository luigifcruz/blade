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

NB_MODULE(_pipeline_impl, m) {
    nb::class_<Pipeline>(m, "pipeline")
        .def(nb::init<const U64&>(), "number_of_streams"_a)
        .def("computeComplete", &Pipeline::computeComplete, nb::rv_policy::reference)
        .def("computeCurrentStepCount", &Pipeline::computeCurrentStepCount, nb::rv_policy::reference)
        .def("computeStepsPerCycle", &Pipeline::computeStepsPerCycle, nb::rv_policy::reference)
        .def("computeLifetimeCycles", &Pipeline::computeLifetimeCycles, nb::rv_policy::reference)
        .def("compute", &Pipeline::compute, nb::rv_policy::reference)
        .def("synchronize", &Pipeline::synchronize, nb::rv_policy::reference)
        .def("isSynchronized", &Pipeline::isSynchronized, nb::rv_policy::reference)
        // TODO: Add stream();
        .def("numberOfStreams", &Pipeline::numberOfStreams, nb::rv_policy::reference)
        .def("__repr__", [](Pipeline& obj){
            return fmt::format("Pipeline()");
        });
}