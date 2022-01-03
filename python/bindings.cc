#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "blade/base.hh"
#include "blade/modules/beamformer/ata.hh"
#include "blade/modules/channelizer.hh"

using namespace Blade;

namespace py = pybind11;

void init_beamformer(const py::module& m) {
    using Class = Modules::Beamformer::ATA<CF32, CF32>;

    py::class_<Class, std::shared_ptr<Class>> beamformer(m, "Beamformer");

    py::class_<Class::Config>(beamformer, "Config")
        .def(py::init<const ArrayDims&, const std::size_t&>(), py::arg("dims"),
                                                               py::arg("block_size"));

    py::class_<Class::Input>(beamformer, "Input")
        .def(py::init<const Vector<Device::CUDA, CF32>&,
                      const Vector<Device::CUDA, CF32>&>(), py::arg("buf"),
                                                            py::arg("phasors"));

    beamformer
        .def(py::init<const Class::Config&,
                      const Class::Input&>(), py::arg("config"),
                                              py::arg("input"))
        .def("inputSize", &Class::getInputSize)
        .def("outputSize", &Class::getOutputSize)
        .def("phasorSize", &Class::getPhasorsSize)
        .def("input", &Class::getInput, py::return_value_policy::reference)
        .def("phasor", &Class::getPhasors, py::return_value_policy::reference)
        .def("output", &Class::getOutput, py::return_value_policy::reference);
}

void init_channelizer(const py::module& m) {
    using Class = Modules::Channelizer<CF32, CF32>;

    py::class_<Class, std::shared_ptr<Class>> channelizer(m, "Channelizer");

    py::class_<Class::Config>(channelizer, "Config")
        .def(py::init<const ArrayDims&,
                      const std::size_t&,
                      const std::size_t&>(), py::arg("dims"),
                                             py::arg("fft_size"),
                                             py::arg("block_size"));

    py::class_<Class::Input>(channelizer, "Input")
        .def(py::init<const Vector<Device::CUDA, CF32>&>(), py::arg("buf"));

    channelizer
        .def(py::init<const Class::Config&,
                      const Class::Input&>(), py::arg("config"),
                                              py::arg("input"))
        .def("bufferSize", &Class::getBufferSize)
        .def("outputDims", &Class::getOutputDims)
        .def("input", &Class::getInput, py::return_value_policy::reference)
        .def("output", &Class::getOutput, py::return_value_policy::reference);
}

class PipelinePub : public Pipeline {
 public:
    using Pipeline::compute;
    using Pipeline::copy;
    using Pipeline::connect;
};

template<typename T>
void init_pipeline_connect(auto& m) {
    m.def("connect", [](PipelinePub& obj, const typename T::Config& config,
                                          const typename T::Input& input){
        auto module = std::shared_ptr<T>();
        obj.connect(module, config, input);
        return module;
    }, py::arg("config"), py::arg("input"));
}

template<typename T>
void init_pipeline_copy(auto& m) {
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

void init_pipeline(const py::module& m) {
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

template<Device D, typename T>
void init_vector(const py::module& m, const char* type) {
    using Class = Vector<D, T>;

    py::class_<Class, std::shared_ptr<Class>>(m, type, py::buffer_protocol())
        .def(py::init<>())
        .def(py::init<const std::size_t&>())
        .def_buffer([](Class& obj){
            return py::buffer_info(obj.data(), sizeof(T),
                                   py::format_descriptor<T>::format(), obj.size());
        })
        .def("__getitem__", [](Class& obj, const std::size_t& index){
            return obj[index];
        }, py::return_value_policy::reference)
        .def("__setitem__", [](Class& obj, const std::size_t& index, const T& val){
            obj[index] = val;
        })
        .def("__len__", [](Class& obj){
            return obj.size();
        });
}

PYBIND11_MODULE(blade, m) {
    static Logger logger{};

    m.def("BL_CHECK", [](const Result& result) {
        BL_CHECK_THROW(result);
    });

    py::enum_<Result>(m, "Result")
        .value("SUCCESS", Result::SUCCESS)
        .value("ERROR", Result::ERROR)
        .value("CUDA_ERROR", Result::CUDA_ERROR)
        .value("ASSERTION_ERROR", Result::ASSERTION_ERROR);

    py::class_<ArrayDims>(m, "ArrayDims")
        .def(py::init<const std::size_t&,
                      const std::size_t&,
                      const std::size_t&,
                      const std::size_t&,
                      const std::size_t&>(), py::arg("NBEAMS"),
                                             py::arg("NANTS"),
                                             py::arg("NCHANS"),
                                             py::arg("NTIME"),
                                             py::arg("NPOLS"))
        .def_property_readonly("NBEAMS", [](ArrayDims& obj){
            return obj.NBEAMS;
        })
        .def_property_readonly("NANTS", [](ArrayDims& obj){
            return obj.NANTS;
        })
        .def_property_readonly("NCHANS", [](ArrayDims& obj){
            return obj.NCHANS;
        })
        .def_property_readonly("NTIME", [](ArrayDims& obj){
            return obj.NTIME;
        })
        .def_property_readonly("NPOLS", [](ArrayDims& obj){
            return obj.NPOLS;
        })
        .def("size", [](ArrayDims& obj){
            return obj.getSize();
        });

    py::module vector = m.def_submodule("vector");

    py::module cpu = vector.def_submodule("cpu");
    init_vector<Device::CPU, CF32>(cpu, "cf32");
    init_vector<Device::CPU, F32>(cpu, "f32");

    py::module cuda = vector.def_submodule("cuda");
    init_vector<Device::CUDA, CF32>(cuda, "cf32");
    init_vector<Device::CUDA, F32>(cuda, "f32");

    init_beamformer(m);
    init_channelizer(m);
    init_pipeline(m);
}
