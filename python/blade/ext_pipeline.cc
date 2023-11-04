#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

#include "blade/base.hh"

namespace nb = nanobind;
using namespace nb::literals;
using namespace Blade;

NB_MODULE(_pipeline_impl, m) {
    nb::class_<Pipeline>(m, "pipeline")
        .def(nb::init<>())
        .def("compute_complete", &Pipeline::computeComplete)
        .def("compute_current_step_count", &Pipeline::computeCurrentStepCount, nb::rv_policy::reference)
        .def("compute_steps_per_cycle", &Pipeline::computeStepsPerCycle, nb::rv_policy::reference)
        .def("compute_lifetime_cycles", &Pipeline::computeLifetimeCycles, nb::rv_policy::reference)
        .def("compute", &Pipeline::compute)
        .def("synchronize", &Pipeline::synchronize)
        .def("commited", &Pipeline::commited, nb::rv_policy::reference)
        .def("is_synchronized", &Pipeline::isSynchronized)
        .def("stream", &Pipeline::stream, "index"_a = 0, nb::rv_policy::reference)
        .def("number_of_streams", &Pipeline::numberOfStreams)
        .def("add_module", &Pipeline::addModule)
        .def("will_output", &Pipeline::willOutput)
        .def("__repr__", [](Pipeline& obj){
            return fmt::format("Pipeline(current_step_count={}, steps_per_cycle={}, lifetime_cycles={})",
                            obj.computeCurrentStepCount(), obj.computeStepsPerCycle(), obj.computeLifetimeCycles());
        });
}
