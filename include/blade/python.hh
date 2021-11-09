#ifndef BLADE_PYTHON_H
#define BLADE_PYTHON_H

#include <pybind11/embed.h>
#include <pybind11/numpy.h>

#include "blade/common.hh"

namespace py = pybind11;

namespace Blade {

class python {
 protected:
    // WARNING: Interpreter should be destructed last!
    py::scoped_interpreter guard{};
    //

    py::object lib;

    template<typename IT, typename OT>
    static std::span<OT> getVector(const py::object& input) {
        auto arr = input().cast<py::array_t<IT>>();
        auto buf = const_cast<OT*>(arr.data());
        auto len = static_cast<std::size_t>(arr.size());
        return std::span{buf, len};
    }
};

}  // namespace Blade

#endif  // BLADE_INCLUDE_BLADE_PYTHON_HH_
