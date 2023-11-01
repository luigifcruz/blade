#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include "blade/base.hh"
#include "blade/modules/base.hh"

namespace nb = nanobind;
using namespace nb::literals;
using namespace Blade;

template<template<typename...> class Base, 
         template<typename...> class Generic, 
         typename OT>
void NB_SUBMODULE_TELESCOPE(auto& m, const auto& tel_name, const auto& out_name) {
    using Class = Base<OT>;
    using GenericClass = Generic<OT>;

    auto mm = m.def_submodule(tel_name)
               .def_submodule(out_name);

    nb::class_<Class, GenericClass> mod(mm, "mod");

    mod
        .def(nb::init<const typename GenericClass::Config&,
                      const typename GenericClass::Input&,
                      const Stream&>(), "config"_a,
                                        "input"_a,
                                        "stream"_a);
}

template<template<typename...> class Generic, 
         typename OT>
void NB_SUBMODULE_GENERIC(auto& m, const auto& out_name) {
    using GenericClass = Generic<OT>;

    auto mm = m.def_submodule(out_name);

    nb::class_<GenericClass, Module> mod(mm, "_generic_mod");

    nb::class_<typename GenericClass::Config>(mod, "config")
        .def(nb::init<const U64&,
                      const U64&,
                      const U64&,
                      const F64&,
                      const F64&,
                      const F64&,
                      const U64&,
                      const U64&,
                      const LLA&,
                      const RA_DEC&,
                      const std::vector<XYZ>&,
                      const ArrayTensor<Device::CPU, CF64>&,
                      const std::vector<RA_DEC>&,
                      const U64&>(), "number_of_antennas"_a,
                                     "number_of_frequency_channels"_a,
                                     "number_of_polarizations"_a,
                                     "observation_frequency_hz"_a,
                                     "channel_bandwidth_hz"_a,
                                     "total_bandwidth_hz"_a,
                                     "frequency_start_index"_a,
                                     "reference_antenna_index"_a,
                                     "array_reference_position"_a,
                                     "boresight_coordinate"_a,
                                     "antenna_positions"_a,
                                     "antenna_calibrations"_a,
                                     "beam_coordinates"_a,
                                     "block_size"_a = 512);

    nb::class_<typename GenericClass::Input>(mod, "input")
        .def(nb::init<const Tensor<Device::CPU, F64>&,
                      const Tensor<Device::CPU, F64>&>(), "block_julian_date"_a,
                                                          "block_dut"_a);

    mod
        .def("process", [](GenericClass& instance, const U64& counter) {
            return instance.process(counter);
        })
        .def("get_config", &GenericClass::getConfig, nb::rv_policy::reference)
        .def("get_delays", &GenericClass::getOutputDelays, nb::rv_policy::reference)
        .def("get_phasors", &GenericClass::getOutputPhasors, nb::rv_policy::reference)
        .def("__repr__", [](GenericClass& obj){
            return fmt::format("Phasor()");
        });
}

NB_MODULE(_phasor_impl, m) {
    // Generic classes.

    NB_SUBMODULE_GENERIC<Modules::Phasor::Generic, CF32>(m, "out_cf32");
    NB_SUBMODULE_GENERIC<Modules::Phasor::Generic, CF64>(m, "out_cf64");

    // Telescope specific classes.

#ifdef BLADE_MODULE_ATA_PHASOR
    NB_SUBMODULE_TELESCOPE<Modules::Phasor::ATA, 
                           Modules::Phasor::Generic, 
                           CF32>(m, "tel_ata", "out_cf32");
    NB_SUBMODULE_TELESCOPE<Modules::Phasor::ATA,
                           Modules::Phasor::Generic, 
                           CF64>(m, "tel_ata", "out_cf64");
#endif
}
