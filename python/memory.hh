#include "base.hh"

#include <blade/base.hh>

#include <memory>

using namespace Blade;
namespace py = pybind11;

template<Device Dev, typename Type, class Shape>
inline void init_vector(const py::module& m, const char* typeName) {
    using Class = Vector<Dev, Type, Shape>;

    py::class_<Class, std::shared_ptr<Class>>(m, typeName, py::buffer_protocol())
        .def(py::init<>())
        .def(py::init<const Shape&, const bool&>(), py::arg("shape"),
                                                    py::arg("unified") = false)
        .def(py::init<const typename Shape::Type&, const bool&>(), py::arg("shape"), 
                                                                   py::arg("unified") = false)
        .def("asnumpy", [](Class& obj){
            std::vector<I64> strides;
            for (U64 i = 0; i < obj.shape().dimensions(); i++) {
                typename Shape::Type shape({0});
                shape[i] = 1;
                strides.push_back(obj.shape().shapeToOffset(shape) * sizeof(Type));
            }

            std::vector<I64> shape;
            for (U64 i = 0; i < obj.shape().dimensions(); ++i) {
                shape.push_back(static_cast<I64>(obj.shape()[i]));
            }

            return py::array_t<Type>(
                shape,                                   /* Buffer dimensions */
                strides,                                 /* Strides (in bytes) for each index */
                obj.data(),                              /* Pointer to buffer */
                py::capsule(new Class(obj), [](void* p){ delete reinterpret_cast<Class*>(p); })
             );
        }, py::return_value_policy::reference)
        .def("__getitem__", [](Class& obj, const typename Shape::Type& shape){
            return obj[shape];
        }, py::return_value_policy::reference)
        .def("__getitem__", [](Class& obj, const U64& index){
            return obj[index];
        }, py::return_value_policy::reference)
        .def("__setitem__", [](Class& obj, const typename Shape::Type& shape, const Type& val){
            obj[shape] = val;
        })
        .def("__setitem__", [](Class& obj, const U64& index, const Type& val){
            obj[index] = val;
        })
        .def("__str__", [](Class& obj){
            return py::str(fmt::format("Vector(shape={}, unified={}, hash={})",
                                       obj.shape(), obj.unified(), obj.hash()));
        })
        .def("unified", [](Class& obj){
            return obj.unified();
        })
        .def("hash", [](Class& obj){
            return obj.hash();
        })
        .def("shape", [](Class& obj) {
            return obj.shape();
        });
}

template<Device Dev, typename Type>
inline void init_vector_dims(py::module& m, const char* name) {
    py::module sub = m.def_submodule(name);
    init_vector<Dev, Type, ArrayShape>(sub, "ArrayTensor");
    init_vector<Dev, Type, PhasorShape>(sub, "PhasorTensor");
    init_vector<Dev, Type, VectorShape>(sub, "Tensor");
}

template<Device Dev>
inline void init_vector_device(py::module& m, const char* name) {
    py::module sub = m.def_submodule(name);
    init_vector_dims<Dev, CF32>(sub, "cf32");
    init_vector_dims<Dev, CF64>(sub, "cf64");
    init_vector_dims<Dev, F32>(sub, "f32");
    init_vector_dims<Dev, F64>(sub, "f64");
}

inline void init_memory_vector(py::module& m) {
    py::module vector = m;

    py::class_<ArrayShape>(vector, "ArrayShape")
        .def(py::init<const typename ArrayShape::Type&>(), py::arg("shape"))
        .def("numberOfAspects", [](ArrayShape& obj){
            return obj.numberOfAspects();
        })
        .def("numberOfFrequencyCHannels", [](ArrayShape& obj){
            return obj.numberOfFrequencyChannels();
        })
        .def("numberOfTimeSamples", [](ArrayShape& obj){
            return obj.numberOfTimeSamples();
        })
        .def("numberOfPolarizations", [](ArrayShape& obj){
            return obj.numberOfPolarizations();
        })
        .def("__getitem__", [](ArrayShape& obj, const U64& index){
            return obj[index];
        }, py::return_value_policy::reference)
        .def("__str__", [](ArrayShape& obj){
            return py::str(fmt::format("ArrayShape(shape={})", obj));
        })
        .def("__len__", [](ArrayShape& obj){
            return obj.size();
        });

    py::class_<PhasorShape>(vector, "PhasorShape")
        .def(py::init<const typename PhasorShape::Type&>(), py::arg("shape"))
        .def("numberOfBeams", [](PhasorShape& obj){
            return obj.numberOfBeams();
        })
        .def("numberOfAntennas", [](PhasorShape& obj){
            return obj.numberOfAntennas();
        })
        .def("numberOfFrequencyCHannels", [](PhasorShape& obj){
            return obj.numberOfFrequencyChannels();
        })
        .def("numberOfTimeSamples", [](PhasorShape& obj){
            return obj.numberOfTimeSamples();
        })
        .def("numberOfPolarizations", [](PhasorShape& obj){
            return obj.numberOfPolarizations();
        })
        .def("__getitem__", [](PhasorShape& obj, const U64& index){
            return obj[index];
        }, py::return_value_policy::reference)
        .def("__str__", [](PhasorShape& obj){
            return py::str(fmt::format("PhasorShape(shape={})", obj));
        })
        .def("__len__", [](PhasorShape& obj){
            return obj.size();
        });

    py::class_<VectorShape>(vector, "VectorShape")
        .def(py::init<const typename VectorShape::Type&>(), py::arg("shape"))
        .def("__getitem__", [](VectorShape& obj, const U64& index){
            return obj[index];
        }, py::return_value_policy::reference)
        .def("__str__", [](VectorShape& obj){
            return py::str(fmt::format("VectorShape(shape={})", obj));
        })
        .def("__len__", [](VectorShape& obj){
            return obj.size();
        });

    init_vector_device<Device::CPU>(vector, "cpu");
    init_vector_device<Device::CUDA>(vector, "cuda");
}

inline void init_memory(py::module& m) {
    init_memory_vector(m);
}
