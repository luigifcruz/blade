#include <nanobind/nanobind.h>

#include "blade/base.hh"
#include "blade/memory/custom.hh"

namespace nb = nanobind;
using namespace nb::literals;
using namespace Blade;

template<typename DataType, typename ShapeType>
void NB_SUBMODULE_COPY_TYPE_SHAPE(auto& m) {
    using CpuTensor = Vector<Device::CPU, DataType, ShapeType>;
    using CudaTensor = Vector<Device::CUDA, DataType, ShapeType>;

    m.def("copy", [](CpuTensor& dst, const CpuTensor& src){
        return Copy(dst, src);
    }, "dst"_a, "src"_a);

    m.def("copy", [](CpuTensor& dst, const CudaTensor& src){
        return Copy(dst, src);
    }, "dst"_a, "src"_a);

    m.def("copy", [](CudaTensor& dst, const CpuTensor& src){
        return Copy(dst, src);
    }, "dst"_a, "src"_a);

    m.def("copy", [](CudaTensor& dst, const CudaTensor& src){
        return Copy(dst, src);
    }, "dst"_a, "src"_a);
}

template<typename DataType>
void NB_SUBMODULE_COPY_TYPE(auto& m) {
    NB_SUBMODULE_COPY_TYPE_SHAPE<DataType, ArrayShape>(m);
    NB_SUBMODULE_COPY_TYPE_SHAPE<DataType, PhasorShape>(m);
    NB_SUBMODULE_COPY_TYPE_SHAPE<DataType, VectorShape>(m);
}

void NB_SUBMODULE_COPY(auto& m) {
    NB_SUBMODULE_COPY_TYPE<I8>(m);
    NB_SUBMODULE_COPY_TYPE<F16>(m);
    NB_SUBMODULE_COPY_TYPE<F32>(m);

    NB_SUBMODULE_COPY_TYPE<CI8>(m);
    NB_SUBMODULE_COPY_TYPE<CF16>(m);
    NB_SUBMODULE_COPY_TYPE<CF32>(m);
}

NB_MODULE(_copy_impl, m) {
    NB_SUBMODULE_COPY(m);
}