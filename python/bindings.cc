#pragma GCC diagnostic ignored "-Wterminate"

#include <fmt/core.h>
#include <pybind11/pybind11.h>
#include <pybind11/eval.h>

#define BL_FATAL(...) throw pybind11::value_error( \
fmt::format("[{}@{}] {}", __FILE__, __LINE__, __VA_ARGS__).c_str());

#include "types.hh"
#include "memory.hh"
#include "pipeline.hh"
#include "modules.hh"

PYBIND11_MODULE(blade, m) {
    static Blade::Logger logger{};

    init_types(m);
    init_memory(m);
    init_pipeline(m);
    init_modules(m);
}
