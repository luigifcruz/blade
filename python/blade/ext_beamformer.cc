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

template<template<typename...> class BaseClass, typename IT, typename OT>
void NB_SUBMODULE(auto& m, const auto& name) {
    using Class = BaseClass<IT, OT>;

    nb::class_<Class> mod(m, name);

    nb::class_<typename Class::Config>(mod, "config")
        .def(nb::init<const BOOL&,
                      const BOOL&,
                      const U64&>(),
                                     "enable_incoherent_beam"_a,
                                     "enable_incoherent_beam_sqrt"_a,
                                     "block_size"_a);

    nb::class_<typename Class::Input>(mod, "input")
        .def(nb::init<const ArrayTensor<Device::CUDA, IT>&,
                      const PhasorTensor<Device::CUDA, IT>&>(), "buf"_a,
                                                                "phasors"_a);

    mod
        .def(nb::init<const typename Class::Config&, const typename Class::Input&>())
        .def("process", [](Class& instance, const U64& counter) {
            return instance.process(counter);
        })
        .def("get_config", &Class::getConfig, nb::rv_policy::reference)
        .def("get_input_buffer", &Class::getInputBuffer, nb::rv_policy::reference)
        .def("get_input_phasors", &Class::getInputPhasors, nb::rv_policy::reference)
        .def("get_output", &Class::getOutputBuffer, nb::rv_policy::reference)
        .def("__repr__", [](Class& obj){
            return fmt::format("Beamformer()");
        });
}

NB_MODULE(_beamformer_impl, m) {
#ifdef BLADE_MODULE_ATA_BEAMFORMER
    {
        auto mm = m.def_submodule("ata");
        NB_SUBMODULE<Modules::Beamformer::ATA, CF32, CF32>(mm, "cf32");
    }
#endif
#ifdef BLADE_MODULE_MEERKAT_BEAMFORMER
    {
        auto mm = m.def_submodule("meerkat");
        NB_SUBMODULE<Modules::Beamformer::MeerKAT, CF32, CF32>(mm, "cf32");
    }
#endif
}