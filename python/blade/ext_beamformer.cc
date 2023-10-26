#include <nanobind/nanobind.h>

#include "blade/base.hh"
#include "blade/modules/base.hh"

namespace nb = nanobind;
using namespace nb::literals;
using namespace Blade;

template<template<typename...> class Base, 
         template<typename...> class Generic, 
         typename IT, 
         typename OT>
void NB_SUBMODULE_TELESCOPE(auto& m, const auto& tel_name, const auto& in_name, const auto& out_name) {
    using Class = Base<IT, OT>;
    using GenericClass = Generic<IT, OT>;

    auto mm = m.def_submodule(tel_name)
               .def_submodule(in_name)
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
         typename IT, 
         typename OT>
void NB_SUBMODULE_GENERIC(auto& m, const auto& in_name, const auto& out_name) {
    using GenericClass = Generic<IT, OT>;

    auto mm = m.def_submodule(in_name)
               .def_submodule(out_name);

    nb::class_<GenericClass, Module> mod(mm, "_generic_mod");

    nb::class_<typename GenericClass::Config>(mod, "config")
        .def(nb::init<const BOOL&,
                      const BOOL&,
                      const U64&>(),
                                     "enable_incoherent_beam"_a,
                                     "enable_incoherent_beam_sqrt"_a,
                                     "block_size"_a = 512);

    nb::class_<typename GenericClass::Input>(mod, "input")
        .def(nb::init<const ArrayTensor<Device::CUDA, IT>&,
                      const PhasorTensor<Device::CUDA, IT>&>(), "buffer"_a,
                                                                "phasors"_a);

    mod
        .def("process", [](GenericClass& instance, const U64& counter) {
            return instance.process(counter);
        })
        .def("get_config", &GenericClass::getConfig, nb::rv_policy::reference)
        .def("get_input_buffer", &GenericClass::getInputBuffer, nb::rv_policy::reference)
        .def("get_input_phasors", &GenericClass::getInputPhasors, nb::rv_policy::reference)
        .def("get_output", &GenericClass::getOutputBuffer, nb::rv_policy::reference)
        .def("__repr__", [](GenericClass& obj){
            return fmt::format("Beamformer()");
        });
}

NB_MODULE(_beamformer_impl, m) {
    // Generic classes.

    NB_SUBMODULE_GENERIC<Modules::Beamformer::Generic, CF32, CF32>(m, "in_cf32", "out_cf32");

    // Telescope specific classes.

#ifdef BLADE_MODULE_ATA_BEAMFORMER
    NB_SUBMODULE_TELESCOPE<Modules::Beamformer::ATA, 
                           Modules::Beamformer::Generic, 
                           CF32, 
                           CF32>(m, "tel_ata", "in_cf32", "out_cf32");
#endif
#ifdef BLADE_MODULE_MEERKAT_BEAMFORMER
    NB_SUBMODULE_TELESCOPE<Modules::Beamformer::MeerKAT, 
                           Modules::Beamformer::Generic,
                           CF32,
                           CF32>(m, "tel_meerkat", "in_cf32", "out_cf32");
#endif
#ifdef BLADE_MODULE_VLA_BEAMFORMER
    // TODO: Add VLA beamformer.
#endif
}
