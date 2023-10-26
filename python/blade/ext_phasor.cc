#include <nanobind/nanobind.h>

#include "blade/base.hh"
#include "blade/modules/base.hh"

namespace nb = nanobind;
using namespace nb::literals;
using namespace Blade;

template<template<typename...> class BaseClass, typename OT>
void NB_SUBMODULE(auto& m, const auto& name) {
    using Class = BaseClass<OT>;

    nb::class_<Class, Module> mod(m, name);

    nb::class_<typename Class::Config>(mod, "config")
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

    nb::class_<typename Class::Input>(mod, "input")
        .def(nb::init<const Tensor<Device::CPU, F64>&,
                      const Tensor<Device::CPU, F64>&>(), "block_julian_date"_a,
                                                          "block_dut"_a);

    mod
        .def(nb::init<const typename Class::Config&,
                      const typename Class::Input&,
                      const Stream&>(), "config"_a,
                                        "input"_a,
                                        "stream"_a)
        .def("process", [](Class& instance, const U64& counter) {
            return instance.process(counter);
        })
        .def("get_config", &Class::getConfig, nb::rv_policy::reference)
        .def("get_delays", &Class::getOutputDelays, nb::rv_policy::reference)
        .def("get_phasors", &Class::getOutputPhasors, nb::rv_policy::reference)
        .def("__repr__", [](Class& obj){
            return fmt::format("Phasor()");
        });
}

NB_MODULE(_phasor_impl, m) {
#ifdef BLADE_MODULE_ATA_PHASOR
    {
        auto mm = m.def_submodule("tel_ata");
        NB_SUBMODULE<Modules::Phasor::ATA, CF32>(m, "type_cf32");
        NB_SUBMODULE<Modules::Phasor::ATA, CF64>(m, "type_cf64");
    }
#endif
}
