#include <nanobind/nanobind.h>

#include "blade/base.hh"
#include "blade/bundles/base.hh"
#include "blade/memory/custom.hh"

namespace nb = nanobind;
using namespace nb::literals;
using namespace Blade;

template<typename IT, typename OT>
void NB_SUBMODULE(auto& m, const auto& name) {
    using Class = Bundles::Generic::ModeH<IT, OT>;

    nb::class_<Class, Bundle> mod(m, name);

    nb::class_<typename Class::Config>(mod, "config")
        .def(nb::init<const ArrayShape&,
                      const ArrayShape&,

                      const BOOL&,

                      const U64&,
                      const U64&,

                      const U64&,
                      const U64&,
                      const U64&,
                      const U64&>(),
                                     "input_shape"_a,
                                     "output_shape"_a,

                                     "polarizer_convert_to_circular"_a,

                                     "detector_integration_size"_a,
                                     "detector_number_of_output_polarizations"_a,

                                     "cast_block_size"_a = 512,
                                     "polarizer_block_size"_a = 512,
                                     "channelizer_block_size"_a = 512,
                                     "detector_block_size"_a = 512);

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
            return fmt::format("ModeH(telescope=bl.generic)");
        });
}

NB_MODULE(_modeh_impl, m) {
    NB_SUBMODULE<CF32, CF32>(m, "type_cf32");
}
