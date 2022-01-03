#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <blade/base.hh>
#include <blade/modules/beamformer/ata.hh>
#include <blade/modules/channelizer.hh>

#include <memory>

using namespace Blade;
namespace py = pybind11;

inline void init_beamformer(const py::module& m) {
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

inline void init_channelizer(const py::module& m) {
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

inline void init_modules(const py::module& m) {
    init_channelizer(m);
    init_beamformer(m);
}
