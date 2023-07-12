#include "base.hh"

#include <blade/base.hh>
#include <blade/modules/base.hh>

#include <memory>

using namespace Blade;
namespace py = pybind11;

class PipelinePub : public Pipeline {
 public:
    using Pipeline::compute;
    using Pipeline::connect;
    using Pipeline::getCudaStream;
};

template<typename T>
inline void init_pipeline_connect(auto& m) {
    m.def("connect", [](PipelinePub& obj, const typename T::Config& config,
                                          const typename T::Input& input){
        auto module = std::shared_ptr<T>();
        obj.connect(module, config, input);
        return module;
    }, py::arg("config"), py::arg("input"));
}

template<typename T>
inline void init_pipeline_copy_phasor_tensor(auto& m) {
    m.def("copy", [](PipelinePub& obj, PhasorTensor<Device::CPU, T>& dst,
                                       const PhasorTensor<Device::CPU, T>& src){
        return Memory::Copy(dst, src, obj.getCudaStream());
    }, py::arg("dst"), py::arg("src"));

    m.def("copy", [](PipelinePub& obj, PhasorTensor<Device::CPU, T>& dst,
                                       const PhasorTensor<Device::CUDA, T>& src){
        return Memory::Copy(dst, src, obj.getCudaStream());
    }, py::arg("dst"), py::arg("src"));

    m.def("copy", [](PipelinePub& obj, PhasorTensor<Device::CUDA, T>& dst,
                                       const PhasorTensor<Device::CPU, T>& src){
        return Memory::Copy(dst, src, obj.getCudaStream());
    }, py::arg("dst"), py::arg("src"));

    m.def("copy", [](PipelinePub& obj, PhasorTensor<Device::CUDA, T>& dst,
                                       const PhasorTensor<Device::CUDA, T>& src){
        return Memory::Copy(dst, src, obj.getCudaStream());
    }, py::arg("dst"), py::arg("src"));
}

template<typename T>
inline void init_pipeline_copy_array_tensor(auto& m) {
    m.def("copy", [](PipelinePub& obj, ArrayTensor<Device::CPU, T>& dst,
                                       const ArrayTensor<Device::CPU, T>& src){
        return Memory::Copy(dst, src, obj.getCudaStream());
    }, py::arg("dst"), py::arg("src"));

    m.def("copy", [](PipelinePub& obj, ArrayTensor<Device::CPU, T>& dst,
                                       const ArrayTensor<Device::CUDA, T>& src){
        return Memory::Copy(dst, src, obj.getCudaStream());
    }, py::arg("dst"), py::arg("src"));

    m.def("copy", [](PipelinePub& obj, ArrayTensor<Device::CUDA, T>& dst,
                                       const ArrayTensor<Device::CPU, T>& src){
        return Memory::Copy(dst, src, obj.getCudaStream());
    }, py::arg("dst"), py::arg("src"));

    m.def("copy", [](PipelinePub& obj, ArrayTensor<Device::CUDA, T>& dst,
                                       const ArrayTensor<Device::CUDA, T>& src){
        return Memory::Copy(dst, src, obj.getCudaStream());
    }, py::arg("dst"), py::arg("src"));
}

inline void init_pipeline(const py::module& m) {
    py::class_<PipelinePub, std::shared_ptr<PipelinePub>> pipeline(m, "Pipeline");

    pipeline
        .def(py::init<>())
        .def("synchronize", &Pipeline::synchronize)
        .def("is_synchronized", &Pipeline::isSynchronized)
        .def("compute", &PipelinePub::compute);

#ifdef BLADE_MODULE_ATA_BEAMFORMER
    init_pipeline_connect<Modules::Beamformer::ATA<CF32, CF32>>(pipeline);
#endif
#ifdef BLADE_MODULE_CHANNELIZER
    init_pipeline_connect<Modules::Channelizer<CF32, CF32>>(pipeline);
#endif
#ifdef BLADE_MODULE_DETECTOR
    init_pipeline_connect<Modules::Detector<CF32, F32>>(pipeline);
#endif
#ifdef BLADE_MODULE_POLARIZER
    init_pipeline_connect<Modules::Polarizer<CF32, CF32>>(pipeline);
#endif
#ifdef BLADE_MODULE_ATA_PHASOR
    init_pipeline_connect<Modules::Phasor::ATA<CF32>>(pipeline);
#endif
    init_pipeline_copy_array_tensor<CF32>(pipeline);
    init_pipeline_copy_array_tensor<F32>(pipeline);
    init_pipeline_copy_phasor_tensor<CF32>(pipeline);
    init_pipeline_copy_phasor_tensor<F32>(pipeline);
}
