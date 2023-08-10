#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/shared_ptr.h>

#include "blade/base.hh"
#include "blade/modules/base.hh"
#include "blade/memory/custom.hh"

namespace nb = nanobind;
using namespace nb::literals;
using namespace Blade;

template<class Class>
void NB_SUBMODULE(auto& m, const auto& name) {
    nb::class_<Class> mod(m, name);

    nb::class_<typename Class::Config>(mod, "config")
        .def(nb::init<const std::string&,
                      const U64&,
                      const U64&>(), "filepath"_a,
                                     "channelizer_rate"_a,
                                     "block_size"_a);

    nb::class_<typename Class::Input>(mod, "input");

    mod
        .def(nb::init<const typename Class::Config&, const typename Class::Input&>())
        .def("get_config", &Class::getConfig, nb::rv_policy::reference)
        .def("get_total_shape", &Class::getTotalShape, nb::rv_policy::reference)
        .def("get_reference_position", &Class::getReferencePosition, nb::rv_policy::reference)
        .def("get_boresight_coordinates", &Class::getBoresightCoordinates, nb::rv_policy::reference)
        .def("get_antenna_positions", &Class::getAntennaPositions, nb::rv_policy::reference)
        .def("get_beam_coordinates", &Class::getBeamCoordinates, nb::rv_policy::reference)
        .def("get_antenna_calibrations", &Class::getAntennaCalibrations, nb::rv_policy::reference)
        .def("__repr__", [](Class& obj){
            return fmt::format("Bfr5Reader()");
        });
}

NB_MODULE(_bfr5_impl, m) {
    NB_SUBMODULE<Modules::Bfr5::Reader>(m, "reader");
}