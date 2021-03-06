#include "base.hh"

#include <blade/base.hh>

#include <memory>

using namespace Blade;
namespace py = pybind11;

template<Device D, typename T>
inline void init_vector(const py::module& m, const char* type) {
    using Class = Vector<D, T>;

    py::class_<Class, std::shared_ptr<Class>>(m, type, py::buffer_protocol())
        .def(py::init<>())
        .def(py::init<const U64&>(), py::arg("size"))
        .def_buffer([](Class& obj){
            return py::buffer_info(obj.data(), sizeof(T),
                                   py::format_descriptor<T>::format(), obj.size());
        })
        .def("__getitem__", [](Class& obj, const U64& index){
            return obj[index];
        }, py::return_value_policy::reference)
        .def("__setitem__", [](Class& obj, const U64& index, const T& val){
            obj[index] = val;
        })
        .def("__len__", [](Class& obj){
            return obj.size();
        });
}

inline void init_memory_vector(py::module& m) {
    py::module vector = m.def_submodule("vector");

    py::module cpu = vector.def_submodule("cpu");
    init_vector<Device::CPU, CF32>(cpu, "cf32");
    init_vector<Device::CPU, F32>(cpu, "f32");
    init_vector<Device::CPU, F64>(cpu, "f64");

    py::module cuda = vector.def_submodule("cuda");
    init_vector<Device::CUDA, CF32>(cuda, "cf32");
    init_vector<Device::CUDA, F32>(cuda, "f32");
    init_vector<Device::CUDA, F64>(cuda, "f64");

    py::module unified = vector.def_submodule("unified");
    init_vector<Device::CUDA | Device::CPU, CF32>(unified, "cf32");
    init_vector<Device::CUDA | Device::CPU, F32>(unified, "f32");
    init_vector<Device::CUDA | Device::CPU, F64>(unified, "f64");
}

inline void init_memory(py::module& m) {
    init_memory_vector(m);
}
