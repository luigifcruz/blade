#include <nanobind/nanobind.h>

#include "blade/base.hh"
#include "blade/modules/base.hh"

namespace nb = nanobind;
using namespace nb::literals;
using namespace Blade;

template<typename IT, typename OT>
void NB_SUBMODULE(auto& m, const auto& name) {
    using Class = Modules::Gather<IT, OT>;

    nb::class_<Class, Module> mod(m, name);

    nb::class_<typename Class::Config>(mod, "config")
        .def(nb::init<const U64&,
                      const U64&,
                      const U64&,
                      const U64&>(), "axis"_a,
                                     "multiplier"_a,
                                     "copy_size_threshold"_a,
                                     "block_size"_a = 512);

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
            return fmt::format("Gather()");
        });
}

NB_MODULE(_gather_impl, m) {
    NB_SUBMODULE<CF32, CF32>(m, "type_cf32");
    NB_SUBMODULE<CF16, CF16>(m, "type_cf16");
    NB_SUBMODULE<CI8, CI8>(m, "type_ci8");
}
