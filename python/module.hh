#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

#include "blade/base.hh"
#include "blade/memory/base.hh"
#include "blade/modules/cast.hh"

using namespace Blade;
using namespace Blade::Modules;

namespace py = pybind11;

template<typename Class>
void init_module(py::module& m, const char* name) {
    py::class_<Class, std::shared_ptr<Class>>(m, name)
        .def(py::init([](const std::size_t& inputSize, const std::size_t& blockSize, const Vector<Device::CUDA, CI8>& buffer) {
            return std::make_shared<Class>(
                Cast<CI8, CF32>::Config{
                    .inputSize = inputSize,
                    .blockSize = blockSize,
                },
                Cast<CI8, CF32>::Input{
                    .buf = buffer,
                }
            );
        }));
}

void init_modules(py::module& m) {
    py::module modules = m.def_submodule("modules");

    init_module<Cast<CI8, CF32>>(modules, "Cast_CI8toCF32");
}
