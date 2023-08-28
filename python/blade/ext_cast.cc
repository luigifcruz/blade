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

template<typename IT, typename OT>
void NB_SUBMODULE(auto& m, const auto& name) {
    using Class = Modules::Cast<IT, OT>;

    nb::class_<Class, Module> mod(m, name);

    nb::class_<typename Class::Config>(mod, "config")
        .def(nb::init<const U64&>(), "block_size"_a = 512);

    nb::class_<typename Class::Input>(mod, "input")
        .def(nb::init<const ArrayTensor<Device::CUDA, IT>&>(), "buffer"_a);

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
        .def("get_input", &Class::getInputBuffer, nb::rv_policy::reference)
        .def("get_output", &Class::getOutputBuffer, nb::rv_policy::reference)
        .def("__repr__", [](Class& obj){
            return fmt::format("Cast()");
        });
}

NB_MODULE(_cast_impl, m) {
    NB_SUBMODULE< CI8,  CI8>(m,  "type_ci8");
    NB_SUBMODULE<CF16, CF16>(m, "type_cf16");
    NB_SUBMODULE<CF32, CF32>(m, "type_cf32");
    NB_SUBMODULE<  I8,   I8>(m,   "type_i8");
    NB_SUBMODULE< F16,  F16>(m,  "type_f16");
    NB_SUBMODULE< F32,  F32>(m,  "type_f32");

    NB_SUBMODULE< CI8, CF32>(m, "type_cf32");
    NB_SUBMODULE< CI8, CF16>(m, "type_cf16");
    NB_SUBMODULE<CF16,  F16>(m,  "type_f16");
    NB_SUBMODULE<CF16,  F32>(m,  "type_f32");
    NB_SUBMODULE<CF16, CF32>(m, "type_cf32");
    NB_SUBMODULE<CF32,  F16>(m,  "type_f16");
    NB_SUBMODULE<CF32,  F32>(m,  "type_f32");
    NB_SUBMODULE<CF32, CF16>(m, "type_cf16");
    NB_SUBMODULE< F16,  F32>(m,  "type_f32");
    NB_SUBMODULE< F16, CF32>(m, "type_cf32");
    NB_SUBMODULE< F16, CF16>(m, "type_cf16");
    NB_SUBMODULE< F32,  F16>(m,  "type_f16");
    NB_SUBMODULE< F32, CF32>(m, "type_cf32");
    NB_SUBMODULE< F32, CF16>(m, "type_cf16");
}
