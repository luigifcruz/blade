#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/string.h>

#include "blade/base.hh"
#include "blade/memory/custom.hh"

namespace nb = nanobind;
using namespace nb::literals;
using namespace Blade;

template<Device DeviceType, typename DataType, typename ShapeType>
void NB_SUBMODULE_VECTOR(auto& m, const auto& name) {
    using ClassType = Vector<DeviceType, DataType, ShapeType>;

    auto mm = 
        nb::class_<ClassType>(m, name)
            .def(nb::init<>())
            .def(nb::init<const ShapeType&, const bool&>(), "shape"_a, "unified"_a = false)
            .def(nb::init<const typename ShapeType::Type&, const bool&>(), "shape"_a, "unified"_a = false)
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
                return fmt::format("Vector({}, dtype={}, device={}, unified={}, hash={})",
                                   obj.shape(), obj.type(), obj.device(), obj.unified(), obj.hash());
            })
            .def_prop_ro("device", [](ClassType& obj){
                return obj.device();
            })
            .def_prop_ro("unified", [](ClassType& obj){
                return obj.unified();
            })
            .def_prop_ro("hash", [](ClassType& obj){
                return obj.hash();
            })
            .def_prop_ro("shape", [](ClassType& obj) {
                return obj.shape();
            }, nb::rv_policy::reference);

    nb::class_<Duet<ClassType>>(m, fmt::format("{}_duet", name).c_str())
        .def(nb::init<const typename ShapeType::Type&, const bool&>(), "shape"_a, "unified"_a = false)
        .def("set", &Duet<ClassType>::set)
        .def("at", &Duet<ClassType>::at, nb::rv_policy::reference)
        .def("__getitem__", &Duet<ClassType>::operator[], nb::rv_policy::reference)
        .def("__call__", &Duet<ClassType>::operator ClassType&, nb::rv_policy::reference);

    // TODO: Add support for all formats.
    if constexpr (!std::is_same<F16, DataType>::value &&
                  !std::is_same<CF16, DataType>::value && 
                  !std::is_same<CI8, DataType>::value) {
        mm.def("as_numpy", [](ClassType& obj){
            ClassType* p = new ClassType(obj);
            nb::capsule deleter(p, [](void *p) noexcept {
                delete reinterpret_cast<ClassType*>(p);
            });

            auto* value = p->data();
            const U64* shape = p->shape().data();
            constexpr const U64 ndims = std::tuple_size<typename ShapeType::Type>::value;
            int32_t device_type = (DeviceType == Device::CUDA) ? nb::device::cuda::value : 
                                                                 nb::device::cpu::value;

            return nb::ndarray<nb::numpy, DataType, nb::shape<ndims>>(value, 
                                                                      ndims, 
                                                                      shape, 
                                                                      deleter,
                                                                      nullptr,
                                                                      nb::dtype<DataType>(), 
                                                                      device_type);
        }, nb::rv_policy::reference);
    }
}

template<Device DeviceType, typename DataType>
void NB_SUBMODULE_MEMORY_DEVICE_TYPE(auto& m, const auto& name) {
    auto mm = m.def_submodule(name);

    NB_SUBMODULE_VECTOR<DeviceType, DataType, ArrayShape>(mm, "array_tensor");
    NB_SUBMODULE_VECTOR<DeviceType, DataType, PhasorShape>(mm, "phasor_tensor");
    NB_SUBMODULE_VECTOR<DeviceType, DataType, DelayShape>(mm, "delay_tensor");
    NB_SUBMODULE_VECTOR<DeviceType, DataType, VectorShape>(mm, "tensor");
}

template<Device DeviceType>
void NB_SUBMODULE_MEMORY_DEVICE(auto& m, const auto& name) {
    auto mm = m.def_submodule(name);

    NB_SUBMODULE_MEMORY_DEVICE_TYPE<DeviceType, I8>(mm, "i8");
    NB_SUBMODULE_MEMORY_DEVICE_TYPE<DeviceType, F16>(mm, "f16");
    NB_SUBMODULE_MEMORY_DEVICE_TYPE<DeviceType, F32>(mm, "f32");
    NB_SUBMODULE_MEMORY_DEVICE_TYPE<DeviceType, F64>(mm, "f64");

    NB_SUBMODULE_MEMORY_DEVICE_TYPE<DeviceType, CI8>(mm, "ci8");
    NB_SUBMODULE_MEMORY_DEVICE_TYPE<DeviceType, CF16>(mm, "cf16");
    NB_SUBMODULE_MEMORY_DEVICE_TYPE<DeviceType, CF32>(mm, "cf32");
    NB_SUBMODULE_MEMORY_DEVICE_TYPE<DeviceType, CF64>(mm, "cf64");
}

NB_MODULE(_mem_impl, m) {
    nb::class_<ArrayShape>(m, "array_shape")
        .def(nb::init<const typename ArrayShape::Type&>(), "shape"_a)
        .def_prop_ro("number_of_aspects", [](ArrayShape& obj){
            return obj.numberOfAspects();
        }, nb::rv_policy::reference)
        .def_prop_ro("number_of_frequency_channels", [](ArrayShape& obj){
            return obj.numberOfFrequencyChannels();
        }, nb::rv_policy::reference)
        .def_prop_ro("number_of_time_samples", [](ArrayShape& obj){
            return obj.numberOfTimeSamples();
        }, nb::rv_policy::reference)
        .def_prop_ro("number_of_polarizations", [](ArrayShape& obj){
            return obj.numberOfPolarizations();
        }, nb::rv_policy::reference)
        .def("__getitem__", [](ArrayShape& obj, const U64& index){
            return obj[index];
        }, nb::rv_policy::reference)
        .def("__repr__", [](ArrayShape& obj){
            return fmt::format("ArrayShape(shape={})", obj);
        })
        .def("__len__", [](ArrayShape& obj){
            return obj.size();
        });
    nb::implicitly_convertible<typename ArrayShape::Type, ArrayShape>();

    nb::class_<PhasorShape>(m, "phasor_shape")
        .def(nb::init<const typename PhasorShape::Type&>(), "shape"_a)
        .def_prop_ro("number_of_beams", [](PhasorShape& obj){
            return obj.numberOfBeams();
        }, nb::rv_policy::reference)
        .def_prop_ro("number_of_antennas", [](PhasorShape& obj){
            return obj.numberOfAntennas();
        }, nb::rv_policy::reference)
        .def_prop_ro("number_of_frequency_channels", [](PhasorShape& obj){
            return obj.numberOfFrequencyChannels();
        }, nb::rv_policy::reference)
        .def_prop_ro("number_of_time_samples", [](PhasorShape& obj){
            return obj.numberOfTimeSamples();
        }, nb::rv_policy::reference)
        .def_prop_ro("number_of_polarizations", [](PhasorShape& obj){
            return obj.numberOfPolarizations();
        }, nb::rv_policy::reference)
        .def("__getitem__", [](PhasorShape& obj, const U64& index){
            return obj[index];
        }, nb::rv_policy::reference)
        .def("__repr__", [](PhasorShape& obj){
            return fmt::format("PhasorShape(shape={})", obj);
        })
        .def("__len__", [](PhasorShape& obj){
            return obj.size();
        });
    nb::implicitly_convertible<typename PhasorShape::Type, PhasorShape>();

    nb::class_<VectorShape>(m, "vector_shape")
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
    nb::implicitly_convertible<typename VectorShape::Type, VectorShape>();

    nb::class_<DelayShape>(m, "delay_shape")
        .def(nb::init<const typename DelayShape::Type&>(), "shape"_a)
        .def_prop_ro("number_of_beams", [](DelayShape& obj){
            return obj.numberOfBeams();
        }, nb::rv_policy::reference)
        .def_prop_ro("number_of_antennas", [](DelayShape& obj){
            return obj.numberOfAntennas();
        }, nb::rv_policy::reference)
        .def("__getitem__", [](DelayShape& obj, const U64& index){
            return obj[index];
        }, nb::rv_policy::reference)
        .def("__repr__", [](DelayShape& obj){
            return fmt::format("DelayShape(shape={})", obj);
        })
        .def("__len__", [](DelayShape& obj){
            return obj.size();
        });
    nb::implicitly_convertible<typename DelayShape::Type, DelayShape>();

    NB_SUBMODULE_MEMORY_DEVICE<Device::CPU>(m, "cpu");
    NB_SUBMODULE_MEMORY_DEVICE<Device::CUDA>(m, "cuda");
}
