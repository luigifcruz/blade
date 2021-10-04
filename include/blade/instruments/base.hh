#ifndef BLADE_INSTRUMENTS_H
#define BLADE_INSTRUMENTS_H

#include "blade/base.hh"

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

namespace Blade::Instrument {

class BLADE_API Generic {
protected:
    template<typename IT, typename OT>
    static std::span<OT> __convert(const py::object & input) {
        auto arr = input().cast<py::array_t<IT>>();
        auto buf = reinterpret_cast<OT*>(arr.data());
        auto len = static_cast<std::size_t>(arr.size());
        return std::span{buf, len};
    }
};

} // namespace Blade::Instrument

#endif

