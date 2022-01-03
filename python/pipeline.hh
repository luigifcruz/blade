#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <blade/base.hh>
#include <blade/modules/beamformer/ata.hh>
#include <blade/modules/channelizer.hh>

#include <memory>

using namespace Blade;
namespace py = pybind11;

class PipelinePub : public Pipeline {
 public:
    using Pipeline::compute;
    using Pipeline::copy;
    using Pipeline::connect;
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
inline void init_pipeline_copy(auto& m) {
    m.def("copy", [](PipelinePub& obj, Vector<Device::CPU, T>& dst,
                                       const Vector<Device::CPU, T>& src){
        return obj.copy(dst, src);
    }, py::arg("dst"), py::arg("src"));

    m.def("copy", [](PipelinePub& obj, Vector<Device::CPU, T>& dst,
                                       const Vector<Device::CUDA, T>& src){
        return obj.copy(dst, src);
    }, py::arg("dst"), py::arg("src"));

    m.def("copy", [](PipelinePub& obj, Vector<Device::CUDA, T>& dst,
                                       const Vector<Device::CPU, T>& src){
        return obj.copy(dst, src);
    }, py::arg("dst"), py::arg("src"));

    m.def("copy", [](PipelinePub& obj, Vector<Device::CUDA, T>& dst,
                                       const Vector<Device::CUDA, T>& src){
        return obj.copy(dst, src);
    }, py::arg("dst"), py::arg("src"));
}

inline void init_pipeline(const py::module& m) {
    py::class_<PipelinePub, std::shared_ptr<PipelinePub>> pipeline(m, "Pipeline");

    pipeline
        .def(py::init<>())
        .def("synchronize", &Pipeline::synchronize)
        .def("isSyncronized", &Pipeline::isSyncronized)
        .def("compute", &PipelinePub::compute);

    init_pipeline_connect<Modules::Beamformer::ATA<CF32, CF32>>(pipeline);
    init_pipeline_connect<Modules::Channelizer<CF32, CF32>>(pipeline);
    init_pipeline_copy<CF32>(pipeline);
}
