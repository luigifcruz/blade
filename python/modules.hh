#include "base.hh"

#include <blade/base.hh>
#include <blade/modules/base.hh>

#include <memory>

using namespace Blade;
namespace py = pybind11;

#ifdef BLADE_MODULE_ATA_BEAMFORMER

inline void init_ata_beamformer(const py::module& m) {
    using Class = Modules::Beamformer::ATA<CF32, CF32>;

    py::class_<Class, std::shared_ptr<Class>> beamformer(m, "Beamformer");

    py::class_<Class::Config>(beamformer, "Config")
        .def(py::init<const BOOL&,
                      const BOOL&,
                      const U64&>(),
                                     py::arg("enable_incoherent_beam"),
                                     py::arg("enable_incoherent_beam_sqrt"),
                                     py::arg("block_size"));

    py::class_<Class::Input>(beamformer, "Input")
        .def(py::init<const ArrayTensor<Device::CUDA, CF32>&,
                      const PhasorTensor<Device::CUDA, CF32>&>(), py::arg("buf"),
                                                                  py::arg("phasors"));

    beamformer
        .def(py::init<const Class::Config&,
                      const Class::Input&>(), py::arg("config"),
                                              py::arg("input"))
        .def("input", &Class::getInputBuffer, py::return_value_policy::reference)
        .def("phasors", &Class::getInputPhasors, py::return_value_policy::reference)
        .def("output", &Class::getOutputBuffer, py::return_value_policy::reference);
}

#endif

#ifdef BLADE_MODULE_ATA_PHASOR

inline void init_ata_phasor(const py::module& m) {
    using Class = Modules::Phasor::ATA<CF32>;

    py::class_<Class, std::shared_ptr<Class>> phasor(m, "Phasor");

    py::class_<Class::Config>(phasor, "Config")
        .def(py::init<const U64&,
                      const U64&,
                      const U64&,
                      const F64&,
                      const F64&,
                      const F64&,
                      const U64&,
                      const U64&,
                      const LLA&,
                      const RA_DEC&,
                      const std::vector<XYZ>&,
                      const std::vector<CF64>&,
                      const std::vector<RA_DEC>&,
                      const U64&,
                      const U64&>(), py::arg("number_of_antennas"),
                                     py::arg("number_of_frequency_channels"),
                                     py::arg("number_of_polarizations"),
                                     py::arg("observation_frequency_hz"),
                                     py::arg("channel_bandwidth_hz"),
                                     py::arg("total_bandwidth_hz"),
                                     py::arg("frequency_start_index"),
                                     py::arg("reference_antenna_index"),
                                     py::arg("array_reference_position"),
                                     py::arg("boresight_coordinate"),
                                     py::arg("antenna_positions"),
                                     py::arg("antenna_coefficients"),
                                     py::arg("beam_coordinates"),
                                     py::arg("preBeamformerChannelizerRate"),
                                     py::arg("block_size"));

    py::class_<Class::Input>(phasor, "Input")
        .def(py::init<const Vector<Device::CPU, F64>&,
                      const Vector<Device::CPU, F64>&,
                      const Vector<Device::CPU, U64>&>(), py::arg("block_julian_date"),
                                                          py::arg("block_dut1"),
                                                          py::arg("block_frequency_channel_offset"));

    phasor
        .def(py::init<const Class::Config&,
                      const Class::Input&>(), py::arg("config"),
                                              py::arg("input"))
        .def("delays", &Class::getOutputDelays, py::return_value_policy::reference)
        .def("phasors", &Class::getOutputPhasors, py::return_value_policy::reference);
}

#endif

#ifdef BLADE_MODULE_CHANNELIZER

inline void init_channelizer(const py::module& m) {
    using Class = Modules::Channelizer<CF32, CF32>;

    py::class_<Class, std::shared_ptr<Class>> channelizer(m, "Channelizer");

    py::class_<Class::Config>(channelizer, "Config")
        .def(py::init<const U64&,
                      const U64&>(), py::arg("rate"),
                                     py::arg("block_size"));

    py::class_<Class::Input>(channelizer, "Input")
        .def(py::init<const ArrayTensor<Device::CUDA, CF32>&>(), py::arg("buf"));

    channelizer
        .def(py::init<const Class::Config&,
                      const Class::Input&>(), py::arg("config"),
                                              py::arg("input"))
        .def("input", &Class::getInputBuffer, py::return_value_policy::reference)
        .def("output", &Class::getOutputBuffer, py::return_value_policy::reference);
}

#endif 

#ifdef BLADE_MODULE_DETECTOR

inline void init_detector(const py::module& m) {
    using Class = Modules::Detector<CF32, F32>;

    py::class_<Class, std::shared_ptr<Class>> detector(m, "Detector");

    py::class_<Class::Config>(detector, "Config")
        .def(py::init<const U64&,
                      const DetectorKernel&,
                      const U64&>(), py::arg("integration_size"),
                                     py::arg("kernel"),
                                     py::arg("block_size"));

    py::class_<Class::Input>(detector, "Input")
        .def(py::init<const ArrayTensor<Device::CUDA, CF32>&>(), py::arg("buf"));
        
    detector
        .def(py::init<const Class::Config&,
                      const Class::Input&>(), py::arg("config"),
                                              py::arg("input"))
        .def("input", &Class::getInputBuffer, py::return_value_policy::reference)
        .def("output", &Class::getOutputBuffer, py::return_value_policy::reference);
}

#endif

#ifdef BLADE_MODULE_POLARIZER

inline void init_polarizer(const py::module& m) {
    using Class = Modules::Polarizer<CF32, CF32>;

    py::class_<Class, std::shared_ptr<Class>> polarizer(m, "Polarizer");

    py::enum_<Class::Mode>(polarizer, "Mode")
        .value("BYPASS", Class::Mode::BYPASS)
        .value("XY2LR", Class::Mode::XY2LR)
        .export_values();

    py::class_<Class::Config>(polarizer, "Config")
        .def(py::init<const Class::Mode, 
                      const U64&>(), py::arg("mode"), 
                                     py::arg("block_size"));

    py::class_<Class::Input>(polarizer, "Input")
        .def(py::init<const ArrayTensor<Device::CUDA, CF32>&>(), py::arg("buf"));
        
    polarizer
        .def(py::init<const Class::Config&,
                      const Class::Input&>(), py::arg("config"),
                                              py::arg("input"))
        .def("input", &Class::getInputBuffer, py::return_value_policy::reference)
        .def("output", &Class::getOutputBuffer, py::return_value_policy::reference);
}

#endif

inline void init_modules(const py::module& m) {
#ifdef BLADE_MODULE_ATA_BEAMFORMER
    init_ata_beamformer(m);
#endif
#ifdef BLADE_MODULE_ATA_PHASOR
    init_ata_phasor(m);
#endif
#ifdef BLADE_MODULE_CHANNELIZER
    init_channelizer(m);
#endif
#ifdef BLADE_MODULE_DETECTOR
    init_detector(m);
#endif
#ifdef BLADE_MODULE_POLARIZER
    init_polarizer(m);
#endif
}
