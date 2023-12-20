#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>

#include "blade/base.hh"

namespace nb = nanobind;
using namespace nb::literals;
using namespace Blade;

NB_MODULE(_hidden_impl, m) {
    // Disable leak warnings because they are not reliable with Python <3.12.
    nb::set_leak_warnings(false);

    // Add custom exception handler for Result errors.
    nb::register_exception_translator(
        [](const std::exception_ptr &p, void*) {
            try {
                std::rethrow_exception(p);
            } catch (const Result &e) {
                PyErr_SetString(PyExc_RuntimeError, "Engine returned an error. Check purple/red text above for details.");
            }
        });

    // Add module type.
    nb::class_<Module>(m, "module")
        .def("__repr__", [](Module& obj) {
            return fmt::format("Module()");
        });

    // Add bundle type.
    nb::class_<Bundle>(m, "bundle")
        .def("modules", &Bundle::getModules)
        .def("__repr__", [](Bundle& obj) {
            return fmt::format("Bundle()");
        });
}
