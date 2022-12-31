#include "base.hh"

#include <blade/base.hh>

#include <memory>

using namespace Blade;
namespace py = pybind11;

template<Device Dev, typename Type, class Dims>
inline void init_vector(const py::module& m, const char* type) {
    using Class = Vector<Dev, Type, Dims>;

    py::class_<Class, std::shared_ptr<Class>>(m, type, py::buffer_protocol())
        .def(py::init<>())
        .def(py::init<const Dims&>(), py::arg("dimensions"))
        .def_buffer([](Class& obj){
            return py::buffer_info(obj.data(), sizeof(Type),
                                   py::format_descriptor<Type>::format(), obj.size());
        })
        .def("__getitem__", [](Class& obj, const U64& index){
            return obj[index];
        }, py::return_value_policy::reference)
        .def("__setitem__", [](Class& obj, const U64& index, const Type& val){
            obj[index] = val;
        })
        .def("__len__", [](Class& obj){
            return obj.size();
        })
        .def("dims", [](Class& obj) {
            return obj.dims();
        });
}

template<Device Dev, typename Type>
inline void init_vector_dims(py::module& m, const char* name) {
    py::module sub = m.def_submodule(name);
    init_vector<Dev, Type, ArrayDimensions>(sub, "ArrayTensor");
    init_vector<Dev, Type, PhasorDimensions>(sub, "PhasorTensor");
    init_vector<Dev, Type, Dimensions>(sub, "Vector");
}

template<Device Dev>
inline void init_vector_device(py::module& m, const char* name) {
    py::module sub = m.def_submodule(name);
    init_vector_dims<Dev, CF32>(sub, "cf32");
    init_vector_dims<Dev, CF64>(sub, "cf64");
    init_vector_dims<Dev, F32>(sub, "f32");
    init_vector_dims<Dev, F64>(sub, "f64");
    init_vector_dims<Dev, U32>(sub, "u32");
    init_vector_dims<Dev, U64>(sub, "u64");
}

inline void init_memory_vector(py::module& m) {
    py::module vector = m.def_submodule("vector");

    py::class_<ArrayDimensions>(vector, "ArrayDimensions")
        .def(py::init<const U64&,
                      const U64&,
                      const U64&,
                      const U64&>(), py::arg("A"),
                                     py::arg("F"),
                                     py::arg("T"),
                                     py::arg("P"))
        .def_property_readonly("shape", [](ArrayDimensions& obj) {
            return std::make_tuple(obj.A, obj.F, obj.T, obj.P);
        })
        .def("__len__", [](ArrayDimensions& obj){
            return obj.size();
        });

    py::class_<PhasorDimensions>(vector, "PhasorDimensions")
        .def(py::init<const U64&,
                      const U64&,
                      const U64&,
                      const U64&,
                      const U64&>(), py::arg("B"),
                                     py::arg("A"),
                                     py::arg("F"),
                                     py::arg("T"),
                                     py::arg("P"))
        .def_property_readonly("shape", [](PhasorDimensions& obj) {
            return std::make_tuple(obj.B, obj.A, obj.F, obj.T, obj.P);
        })
        .def("__len__", [](PhasorDimensions& obj){
            return obj.size();
        });

    py::class_<Dimensions>(vector, "Dimensions")
        .def(py::init([](const U64& n1){
            return Dimensions({n1});
        }))
        .def(py::init([](const U64& n1, const U64& n2){
            return Dimensions({n1, n2});
        }))
        .def(py::init([](const U64& n1, const U64& n2, const U64& n3){
            return Dimensions({n1, n2, n3});
        }))
        .def("__len__", [](Dimensions& obj){
            return obj.size();
        });

    init_vector_device<Device::CPU>(vector, "cpu");
    init_vector_device<Device::CUDA>(vector, "cuda");
    init_vector_device<Device::CUDA | Device::CPU>(vector, "unified");
}

inline void init_memory(py::module& m) {
    init_memory_vector(m);
}
