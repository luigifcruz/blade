#pragma GCC diagnostic ignored "-Wterminate"

#include <fmt/core.h>

#include "base.hh"
#include "types.hh"
#include "memory.hh"
#include "pipeline.hh"
#include "modules.hh"

PYBIND11_MODULE(blade, m) {
    init_types(m);
    init_memory(m);
    init_pipeline(m);
    init_modules(m);
}
