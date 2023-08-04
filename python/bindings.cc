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

// TODO: Add exception translation to Python.
// TODO: Add Pipeline.
// TODO: Add Runner.
// TODO: Add Python wrapper.
// TODO: Add types to modules.

void NB_SUBMODULE_CONSTANTS(auto& m) {
    auto mm = m.def_submodule("const");
}

template<Device DeviceType, typename DataType, typename ShapeType>
void NB_SUBMODULE_VECTOR(auto& m, const auto& name) {
    using ClassType = Vector<DeviceType, DataType, ShapeType>;

    nb::class_<ClassType>(m, name)
        .def(nb::init<>())
        .def(nb::init<const ShapeType&, const bool&>(), "shape"_a, "unified"_a = false)
        .def(nb::init<const typename ShapeType::Type&, const bool&>(), "shape"_a, "unified"_a = false)
        .def("as_numpy", [](ClassType& obj){
            // TODO: Should return nb::ndarray<...>
        }, nb::rv_policy::reference)
        .def("__getitem__", [](ClassType& obj, const typename ShapeType::Type& shape){
            return obj[shape];
        }, nb::rv_policy::reference)
        .def("__getitem__", [](ClassType& obj, const U64& index){
            return obj[index];
        }, nb::rv_policy::reference)
        .def("__setitem__", [](ClassType& obj, const typename ShapeType::Type& shape, const DataType& val){
            obj[shape] = val;
        })
        .def("__setitem__", [](ClassType& obj, const U64& index, const DataType& val){
            obj[index] = val;
        })
        .def("__repr__", [](ClassType& obj){
            return fmt::format("Vector(shape={}, unified={}, hash={})",
                                obj.shape(), obj.unified(), obj.hash());
        })
        .def("unified", [](ClassType& obj){
            return obj.unified();
        })
        .def("hash", [](ClassType& obj){
            return obj.hash();
        })
        .def("shape", [](ClassType& obj) {
            return obj.shape();
        });
}

template<Device DeviceType, typename DataType>
void NB_SUBMODULE_MEMORY_DEVICE_TYPE(auto& m, const auto& name) {
    auto mm = m.def_submodule(name);

    NB_SUBMODULE_VECTOR<DeviceType, DataType, ArrayShape>(mm, "array_tensor");
    NB_SUBMODULE_VECTOR<DeviceType, DataType, PhasorShape>(mm, "phasor_tensor");
    NB_SUBMODULE_VECTOR<DeviceType, DataType, VectorShape>(mm, "tensor");
}

template<Device DeviceType>
void NB_SUBMODULE_MEMORY_DEVICE(auto& m, const auto& name) {
    auto mm = m.def_submodule(name);

    NB_SUBMODULE_MEMORY_DEVICE_TYPE<DeviceType, I8>(mm, "i8");
    NB_SUBMODULE_MEMORY_DEVICE_TYPE<DeviceType, F16>(mm, "f16");
    NB_SUBMODULE_MEMORY_DEVICE_TYPE<DeviceType, F32>(mm, "f32");

    NB_SUBMODULE_MEMORY_DEVICE_TYPE<DeviceType, CI8>(mm, "ci8");
    NB_SUBMODULE_MEMORY_DEVICE_TYPE<DeviceType, CF16>(mm, "cf16");
    NB_SUBMODULE_MEMORY_DEVICE_TYPE<DeviceType, CF32>(mm, "cf32");
}

void NB_SUBMODULE_MEMORY(auto& m) {
    auto mm = m.def_submodule("mem");

    nb::class_<ArrayShape>(mm, "array_shape")
        .def(nb::init<const typename ArrayShape::Type&>(), "shape"_a)
        .def("number_of_aspects", [](ArrayShape& obj){
            return obj.numberOfAspects();
        })
        .def("number_of_frequency_channels", [](ArrayShape& obj){
            return obj.numberOfFrequencyChannels();
        })
        .def("number_of_time_samples", [](ArrayShape& obj){
            return obj.numberOfTimeSamples();
        })
        .def("number_of_polarizations", [](ArrayShape& obj){
            return obj.numberOfPolarizations();
        })
        .def("__getitem__", [](ArrayShape& obj, const U64& index){
            return obj[index];
        }, nb::rv_policy::reference)
        .def("__repr__", [](ArrayShape& obj){
            return fmt::format("ArrayShape(shape={})", obj);
        })
        .def("__len__", [](ArrayShape& obj){
            return obj.size();
        });

    nb::class_<PhasorShape>(mm, "phasor_shape")
        .def(nb::init<const typename PhasorShape::Type&>(), "shape"_a)
        .def("number_of_beams", [](PhasorShape& obj){
            return obj.numberOfBeams();
        })
        .def("number_of_antennas", [](PhasorShape& obj){
            return obj.numberOfAntennas();
        })
        .def("number_of_frequency_channels", [](PhasorShape& obj){
            return obj.numberOfFrequencyChannels();
        })
        .def("number_of_time_samples", [](PhasorShape& obj){
            return obj.numberOfTimeSamples();
        })
        .def("number_of_polarizations", [](PhasorShape& obj){
            return obj.numberOfPolarizations();
        })
        .def("__getitem__", [](PhasorShape& obj, const U64& index){
            return obj[index];
        }, nb::rv_policy::reference)
        .def("__repr__", [](PhasorShape& obj){
            return fmt::format("PhasorShape(shape={})", obj);
        })
        .def("__len__", [](PhasorShape& obj){
            return obj.size();
        });

    nb::class_<VectorShape>(mm, "vector_shape")
        .def(nb::init<const typename VectorShape::Type&>(), "shape"_a)
        .def("__getitem__", [](VectorShape& obj, const U64& index){
            return obj[index];
        }, nb::rv_policy::reference)
        .def("__repr__", [](VectorShape& obj){
            return fmt::format("VectorShape(shape={})", obj);
        })
        .def("__len__", [](VectorShape& obj){
            return obj.size();
        });

    NB_SUBMODULE_MEMORY_DEVICE<Device::CPU>(mm, "cpu");
    NB_SUBMODULE_MEMORY_DEVICE<Device::CUDA>(mm, "cuda");
}

#ifdef BLADE_MODULE_CAST
template<typename IT, typename OT>
void NB_SUBMODULE_MODULES_CAST(auto& m) {
    using Class = Modules::Cast<IT, OT>;

    nb::class_<Class> cast(m, "cast");

    nb::class_<typename Class::Config>(cast, "config")
        .def(nb::init<const U64&>(), "block_size"_a);

    nb::class_<typename Class::Input>(cast, "input")
        .def(nb::init<const ArrayTensor<Device::CUDA, IT>&>(), "buf"_a);

    cast
        .def(nb::init<const typename Class::Config&, const typename Class::Input&>())
        .def("process", [](Class& instance, const U64& counter) {
            return instance.process(counter);
        })
        .def("get_input", &Class::getInputBuffer, nb::rv_policy::reference)
        .def("get_output", &Class::getOutputBuffer, nb::rv_policy::reference);
}
#endif
#ifdef BLADE_MODULE_CHANNELIZER
template<typename IT, typename OT>
void NB_SUBMODULE_MODULES_CHANNELIZER(auto& m) {
    using Class = Modules::Channelizer<IT, OT>;

    nb::class_<Class> channelizer(m, "channelizer");

    nb::class_<typename Class::Config>(channelizer, "config")
        .def(nb::init<const U64&,
                      const U64&>(), "rate"_a,
                                     "block_size"_a);

    nb::class_<typename Class::Input>(channelizer, "input")
        .def(nb::init<const ArrayTensor<Device::CUDA, CF32>&>(), "buf"_a);

    channelizer
        .def(nb::init<const typename Class::Config&, const typename Class::Input&>())
        .def("process", [](Class& instance, const U64& counter) {
            return instance.process(counter);
        })
        .def("get_input", &Class::getInputBuffer, nb::rv_policy::reference)
        .def("get_output", &Class::getOutputBuffer, nb::rv_policy::reference);
}
#endif

#ifdef BLADE_MODULE_DETECTOR
template<typename IT, typename OT>
void NB_SUBMODULE_MODULES_DETECTOR(auto& m) {
    using Class = Modules::Detector<IT, OT>;

    nb::class_<Class> detector(m, "detector");

    nb::class_<typename Class::Config>(detector, "config")
        .def(nb::init<const U64&,
                      const U64&,
                      const U64&>(), "integration_size"_a,
                                     "number_of_output_polarizations"_a,
                                     "block_size"_a);

    nb::class_<typename Class::Input>(detector, "input")
        .def(nb::init<const ArrayTensor<Device::CUDA, CF32>&>(), "buf"_a);
        
    detector
        .def(nb::init<const typename Class::Config&, const typename Class::Input&>())
        .def("process", [](Class& instance, const U64& counter) {
            return instance.process(counter);
        })
        .def("get_input", &Class::getInputBuffer, nb::rv_policy::reference)
        .def("get_output", &Class::getOutputBuffer, nb::rv_policy::reference);
}
#endif

#ifdef BLADE_MODULE_BFR5
template<typename IT, typename OT>
void NB_SUBMODULE_MODULES_BFR5(auto& m) {
}
#endif

#ifdef BLADE_MODULE_GUPPI
template<typename IT, typename OT>
void NB_SUBMODULE_MODULES_GUPPI(auto& m) {
}
#endif

#ifdef BLADE_MODULE_POLARIZER
template<typename IT, typename OT>
void NB_SUBMODULE_MODULES_POLARIZER(auto& m) {
    using Class = Modules::Polarizer<IT, OT>;

    nb::class_<Class> polarizer(m, "polarizer");

    nb::enum_<typename Class::Mode>(polarizer, "mode")
        .value("BYPASS", Class::Mode::BYPASS)
        .value("XY2LR", Class::Mode::XY2LR)
        .export_values();

    nb::class_<typename Class::Config>(polarizer, "config")
        .def(nb::init<const typename Class::Mode, 
                      const U64&>(), "mode"_a, 
                                     "block_size"_a);

    nb::class_<typename Class::Input>(polarizer, "input")
        .def(nb::init<const ArrayTensor<Device::CUDA, CF32>&>(), "buf"_a);
        
    polarizer
        .def(nb::init<const typename Class::Config&, const typename Class::Input&>())
        .def("process", [](Class& instance, const U64& counter) {
            return instance.process(counter);
        })
        .def("get_input", &Class::getInputBuffer, nb::rv_policy::reference)
        .def("get_output", &Class::getOutputBuffer, nb::rv_policy::reference);
}
#endif

#ifdef BLADE_MODULE_GATHER
template<typename IT, typename OT>
void NB_SUBMODULE_MODULES_GATHER(auto& m) {
}
#endif

#ifdef BLADE_MODULE_COPY
template<typename IT, typename OT>
void NB_SUBMODULE_MODULES_COPY(auto& m) {
}
#endif

#ifdef BLADE_MODULE_PERMUTATION
template<typename IT, typename OT>
void NB_SUBMODULE_MODULES_PERMUTATION(auto& m) {
}
#endif

#ifdef BLADE_MODULE_ATA_BEAMFORMER
template<typename IT, typename OT>
void NB_SUBMODULE_MODULES_ATA_BEAMFORMER(auto& m) {
    using Class = Modules::Beamformer::ATA<IT, OT>;

    nb::class_<Class> beamformer(m, "beamformer");

    nb::class_<typename Class::Config>(beamformer, "config")
        .def(nb::init<const BOOL&,
                      const BOOL&,
                      const U64&>(),
                                     "enable_incoherent_beam"_a,
                                     "enable_incoherent_beam_sqrt"_a,
                                     "block_size"_a);

    nb::class_<typename Class::Input>(beamformer, "input")
        .def(nb::init<const ArrayTensor<Device::CUDA, CF32>&,
                      const PhasorTensor<Device::CUDA, CF32>&>(), "buf"_a,
                                                                  "phasors"_a);

    beamformer
        .def(nb::init<const typename Class::Config&, const typename Class::Input&>())
        .def("process", [](Class& instance, const U64& counter) {
            return instance.process(counter);
        })
        .def("get_input", &Class::getInputBuffer, nb::rv_policy::reference)
        .def("get_phasors", &Class::getInputPhasors, nb::rv_policy::reference)
        .def("get_output", &Class::getOutputBuffer, nb::rv_policy::reference);
}
#endif

#ifdef BLADE_MODULE_ATA_PHASOR
template<typename OT>
void NB_SUBMODULE_MODULES_ATA_PHASOR(auto& m) {
    using Class = Modules::Phasor::ATA<OT>;

    nb::class_<Class> phasor(m, "phasor");

    nb::class_<typename Class::Config>(phasor, "config")
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
                                     "block_size"_a);

    nb::class_<typename Class::Input>(phasor, "input")
        .def(nb::init<const Tensor<Device::CPU, F64>&,
                      const Tensor<Device::CPU, F64>&>(), "block_julian_date"_a,
                                                          "block_dut"_a);

    phasor
        .def(nb::init<const typename Class::Config&, const typename Class::Input&>())
        .def("process", [](Class& instance, const U64& counter) {
            return instance.process(counter);
        })
        .def("get_delays", &Class::getOutputDelays, nb::rv_policy::reference)
        .def("get_phasors", &Class::getOutputPhasors, nb::rv_policy::reference);
}
#endif

#ifdef BLADE_MODULE_MEERKAT_BEAMFORMER
template<typename IT, typename OT>
void NB_SUBMODULE_MODULES_MEERKAT_BEAMFORMER(auto& m) {
    using Class = Modules::Beamformer::MeerKAT<IT, OT>;

    nb::class_<Class> beamformer(m, "beamformer");

    nb::class_<typename Class::Config>(beamformer, "config")
        .def(nb::init<const BOOL&,
                      const BOOL&,
                      const U64&>(),
                                     "enable_incoherent_beam"_a,
                                     "enable_incoherent_beam_sqrt"_a,
                                     "block_size"_a);

    nb::class_<typename Class::Input>(beamformer, "input")
        .def(nb::init<const ArrayTensor<Device::CUDA, CF32>&,
                      const PhasorTensor<Device::CUDA, CF32>&>(), "buffer"_a,
                                                                  "phasors"_a);

    beamformer
        .def(nb::init<const typename Class::Config&, const typename Class::Input&>())
        .def("process", [](Class& instance, const U64& counter) {
            return instance.process(counter);
        })
        .def("get_input", &Class::getInputBuffer, nb::rv_policy::reference)
        .def("get_phasors", &Class::getInputPhasors, nb::rv_policy::reference)
        .def("get_output", &Class::getOutputBuffer, nb::rv_policy::reference);
}
#endif

#ifdef BLADE_MODULE_VLA_BEAMFORMER
// TODO: Implement VLA beamformer.
#endif

void NB_SUBMODULE_MODULES(auto& m) {
    auto mm = m.def_submodule("mod");

#ifdef BLADE_MODULE_CAST
    NB_SUBMODULE_MODULES_CAST<CF32, CF32>(mm);
#endif
#ifdef BLADE_MODULE_CHANNELIZER
    NB_SUBMODULE_MODULES_CHANNELIZER<CF32, CF32>(mm);
#endif
#ifdef BLADE_MODULE_DETECTOR
    NB_SUBMODULE_MODULES_DETECTOR<CF32, F32>(mm);
#endif
#ifdef BLADE_MODULE_BFR5
    NB_SUBMODULE_MODULES_BFR5<CF32, CF32>(mm);
#endif
#ifdef BLADE_MODULE_GUPPI
    NB_SUBMODULE_MODULES_GUPPI<CF32, CF32>(mm);
#endif
#ifdef BLADE_MODULE_POLARIZER
    NB_SUBMODULE_MODULES_POLARIZER<CF32, CF32>(mm);
#endif
#ifdef BLADE_MODULE_GATHER
    NB_SUBMODULE_MODULES_GATHER<CF32, CF32>(mm);
#endif
#ifdef BLADE_MODULE_COPY
    NB_SUBMODULE_MODULES_COPY<CF32, CF32>(mm);
#endif
#ifdef BLADE_MODULE_PERMUTATION
    NB_SUBMODULE_MODULES_PERMUTATION<CF32, CF32>(mm);
#endif
#ifdef BLADE_BUNDLE_GENERIC_MODE_H
#endif

    {
        auto mmm = mm.def_submodule("ata");

#ifdef BLADE_MODULE_ATA_BEAMFORMER
        NB_SUBMODULE_MODULES_ATA_BEAMFORMER<CF32, CF32>(mmm);
#endif
#ifdef BLADE_MODULE_ATA_PHASOR
        NB_SUBMODULE_MODULES_ATA_PHASOR<CF32>(mmm);
#endif
#ifdef BLADE_BUNDLE_ATA_MODE_B
#endif
    }

    {
        auto mmm = mm.def_submodule("meerkat");

#ifdef BLADE_MODULE_MEERKAT_BEAMFORMER
        NB_SUBMODULE_MODULES_MEERKAT_BEAMFORMER<CF32, CF32>(mmm);
#endif
    }

    {
        auto mmm = mm.def_submodule("vla");

#ifdef BLADE_MODULE_VLA_BEAMFORMER
        NB_SUBMODULE_MODULES_VLA_BEAMFORMER<CF32, CF32>(mmm);
#endif
    }
}

NB_MODULE(blade, m) {
    NB_SUBMODULE_CONSTANTS(m);
    NB_SUBMODULE_MEMORY(m);
    NB_SUBMODULE_MODULES(m);
}