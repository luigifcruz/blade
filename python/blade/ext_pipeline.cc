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
        .def(nb::init<>())
        .def("compute_complete", &Pipeline::computeComplete, nb::rv_policy::reference)
        .def("compute_current_step_count", &Pipeline::computeCurrentStepCount, nb::rv_policy::reference)
        .def("compute_steps_per_cycle", &Pipeline::computeStepsPerCycle, nb::rv_policy::reference)
        .def("compute_lifetime_cycles", &Pipeline::computeLifetimeCycles, nb::rv_policy::reference)
        .def("compute", &Pipeline::compute, nb::rv_policy::reference)
        .def("synchronize", &Pipeline::synchronize, nb::rv_policy::reference)
        .def("commited", &Pipeline::commited)
        .def("is_synchronized", &Pipeline::isSynchronized, nb::rv_policy::reference)
        .def("stream", &Pipeline::stream, "index"_a = 0)
        .def("number_of_streams", &Pipeline::numberOfStreams, nb::rv_policy::reference)
        .def("add_module", &Pipeline::addModule, nb::rv_policy::reference)
        .def("__repr__", [](Pipeline& obj){
            return fmt::format("Pipeline(current_step_count={}, steps_per_cycle={}, lifetime_cycles={})",
                            obj.computeCurrentStepCount(), obj.computeStepsPerCycle(), obj.computeLifetimeCycles());
        });
}
