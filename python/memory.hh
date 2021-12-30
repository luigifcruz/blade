#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

#include "blade/base.hh"
#include "blade/memory/base.hh"

using namespace Blade;

namespace py = pybind11;

template<Device D, typename T>
void init_memory_vector(py::module& m, const char* device, const char* type) {
    using Class = Vector<D, T>;

    py::module mm = m.def_submodule(device);

    py::class_<Class, std::shared_ptr<Class>>(mm, type)
        .def(py::init<>())
        .def(py::init<const std::size_t&>())
        .def("resize", [](Class& self, const std::size_t& size){
            return self.resize(size);
        });
}

void init_memory(py::module& m) {
    py::module vector = m.def_submodule("vector");

    vector.def("zeroes", [](const std::size_t& size, const Device& device){
        switch (device) {
            case Device::CUDA:
            return std::make_shared<Vector<Device::CUDA, CI8>>(size);
        }
    });

    init_memory_vector<Device::CUDA, CI8>(vector, "cuda", "ci8");
    init_memory_vector<Device::CUDA, CF32>(vector, "cuda", "cf32");
    init_memory_vector<Device::CUDA, CF16>(vector, "cuda", "cf16");
}
