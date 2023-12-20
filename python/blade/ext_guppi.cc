#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "blade/base.hh"
#include "blade/modules/base.hh"

namespace nb = nanobind;
using namespace nb::literals;
using namespace Blade;

template<typename OT>
void NB_SUBMODULE_READER(auto& m, const auto& name, const auto& typeName) {
    using Class = Modules::Guppi::Reader<OT>;

    auto mm = m.def_submodule(name);

    nb::class_<Class, Module> mod(mm, typeName);

    nb::class_<typename Class::Config>(mod, "config")
        .def(nb::init<const std::string&,
                      const U64&,
                      const U64&,
                      const U64&,
                      const U64&>(), "filepath"_a,
                                     "step_number_of_time_samples"_a,
                                     "step_number_of_frequency_channels"_a,
                                     "step_number_of_aspects"_a,
                                     "block_size"_a = 512);

    nb::class_<typename Class::Input>(mod, "input");

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
        .def("get_step_output_buffer", &Class::getStepOutputBuffer, nb::rv_policy::reference)
        .def("get_step_output_julian_date", &Class::getStepOutputJulianDate, nb::rv_policy::reference)
        .def("get_step_output_dut", &Class::getStepOutputDut1, nb::rv_policy::reference)
        .def("get_total_output_buffer_shape", &Class::getTotalOutputBufferShape)
        .def("get_step_output_buffer_shape", &Class::getStepOutputBufferShape)
        .def("get_number_of_steps", &Class::getNumberOfSteps)
        .def("get_total_bandwidth", &Class::getTotalBandwidth)
        .def("get_channel_bandwidth", &Class::getChannelBandwidth)
        .def("get_channel_start_index", &Class::getChannelStartIndex)
        .def("get_observation_frequency", &Class::getObservationFrequency)
        .def("__repr__", [](Class& obj){
            return fmt::format("GuppiReader()");
        });
}

template<typename IT>
void NB_SUBMODULE_WRITER(auto& m, const auto& name, const auto& typeName) {
    using Class = Modules::Guppi::Writer<IT>;

    auto mm = m.def_submodule(name);

    nb::class_<Class> mod(mm, typeName);

    nb::class_<typename Class::Config>(mod, "config")
        .def(nb::init<const std::string&,
                      const bool&,
                      const U64&,
                      const U64&>(), "filepath"_a,
                                     "directio"_a,
                                     "input_frequency_batches"_a,
                                     "block_size"_a = 512);

    nb::class_<typename Class::Input>(mod, "input")
        .def(nb::init<const ArrayTensor<Device::CPU, IT>&>(), "buffer"_a);

    mod
        .def(nb::init<const typename Class::Config&, const typename Class::Input&>())
        .def("process", [](Class& instance, const U64& counter) {
            return instance.process(counter);
        })
        .def("get_config", &Class::getConfig, nb::rv_policy::reference)
        .def("get_input", &Class::getInputBuffer, nb::rv_policy::reference)
        .def("header_put", [](Class& instance, const std::string& key, const F64& value){
            return instance.headerPut(key, value);
        })
        .def("header_put", [](Class& instance, const std::string& key, const I64& value){
            return instance.headerPut(key, value);
        })
        .def("__repr__", [](Class& obj){
            return fmt::format("GuppiWriter()");
        });
}

NB_MODULE(_guppi_impl, m) {
    NB_SUBMODULE_READER<CI8>(m, "taint_reader", "type_ci8");
    NB_SUBMODULE_WRITER<CF16>(m, "taint_writer", "type_cf16");
    NB_SUBMODULE_WRITER<CF32>(m, "taint_writer", "type_cf32");
}
