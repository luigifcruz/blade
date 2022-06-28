#include "base.hh"

#include <blade/base.hh>
#include <blade/modules/beamformer/ata.hh>
#include <blade/modules/channelizer.hh>
#include <blade/modules/detector.hh>
#include <blade/modules/phasor/ata.hh>

#include <memory>

using namespace Blade;
namespace py = pybind11;

inline void init_beamformer(const py::module& m) {
    using Class = Modules::Beamformer::ATA<CF32, CF32>;

    py::class_<Class, std::shared_ptr<Class>> beamformer(m, "Beamformer");

    py::class_<Class::Config>(beamformer, "Config")
        .def(py::init<const U64&,
                      const U64&,
                      const U64&,
                      const U64&,
                      const U64&,
                      const BOOL&,
                      const BOOL&,
                      const U64&>(), py::arg("number_of_beams"),
                                     py::arg("number_of_antennas"),
                                     py::arg("number_of_frequency_channels"),
                                     py::arg("number_of_time_samples"),
                                     py::arg("number_of_polarizations"),
                                     py::arg("enable_incoherent_beam"),
                                     py::arg("enable_incoherent_beam_sqrt"),
                                     py::arg("block_size"));

    py::class_<Class::Input>(beamformer, "Input")
        .def(py::init<const Vector<Device::CUDA, CF32>&,
                      const Vector<Device::CUDA, CF32>&>(), py::arg("buf"),
                                                            py::arg("phasors"));

    beamformer
        .def(py::init<const Class::Config&,
                      const Class::Input&>(), py::arg("config"),
                                              py::arg("input"))
        .def("input_size", &Class::getInputSize)
        .def("output_size", &Class::getOutputSize)
        .def("phasors_size", &Class::getPhasorsSize)
        .def("input", &Class::getInput, py::return_value_policy::reference)
        .def("phasors", &Class::getPhasors, py::return_value_policy::reference)
        .def("output", &Class::getOutput, py::return_value_policy::reference);
}

inline void init_phasor(const py::module& m) {
    using Class = Modules::Phasor::ATA<CF32>;

    py::class_<Class, std::shared_ptr<Class>> phasor(m, "Phasor");

    py::class_<Class::Config>(phasor, "Config")
        .def(py::init<const U64&,
                      const U64&,
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
                      const U64&>(), py::arg("number_of_beams"),
                                     py::arg("number_of_antennas"),
                                     py::arg("number_of_frequency_channels"),
                                     py::arg("number_of_polarizations"),
                                     py::arg("rf_frequency_hz"),
                                     py::arg("channel_bandwidth_hz"),
                                     py::arg("total_bandwidth_hz"),
                                     py::arg("frequency_start_index"),
                                     py::arg("reference_antenna_index"),
                                     py::arg("array_reference_position"),
                                     py::arg("boresight_coordinate"),
                                     py::arg("antenna_positions"),
                                     py::arg("antenna_calibrations"),
                                     py::arg("beam_coordinates"),
                                     py::arg("block_size"));

    py::class_<Class::Input>(phasor, "Input")
        .def(py::init<const Vector<Device::CPU, F64>&,
                      const Vector<Device::CPU, F64>&>(), py::arg("frame_julian_date"),
                                                          py::arg("frame_dut1"));

    phasor
        .def(py::init<const Class::Config&,
                      const Class::Input&>(), py::arg("config"),
                                              py::arg("input"))
        .def("phasors_size", &Class::getPhasorsSize)
        .def("delays_size", &Class::getDelaysSize)
        .def("calibration_size", &Class::getCalibrationsSize)
        .def("delays", &Class::getDelays, py::return_value_policy::reference)
        .def("phasors", &Class::getPhasors, py::return_value_policy::reference);
}

inline void init_channelizer(const py::module& m) {
    using Class = Modules::Channelizer<CF32, CF32>;

    py::class_<Class, std::shared_ptr<Class>> channelizer(m, "Channelizer");

    py::class_<Class::Config>(channelizer, "Config")
        .def(py::init<const U64&,
                      const U64&,
                      const U64&,
                      const U64&,
                      const U64&,
                      const U64&,
                      const U64&>(), py::arg("number_of_beams"),
                                     py::arg("number_of_antennas"),
                                     py::arg("number_of_frequency_channels"),
                                     py::arg("number_of_time_samples"),
                                     py::arg("number_of_polarizations"),
                                     py::arg("fft_size"),
                                     py::arg("block_size"));

    py::class_<Class::Input>(channelizer, "Input")
        .def(py::init<const Vector<Device::CUDA, CF32>&>(), py::arg("buf"));

    channelizer
        .def(py::init<const Class::Config&,
                      const Class::Input&>(), py::arg("config"),
                                              py::arg("input"))
        .def("buffer_size", &Class::getBufferSize)
        .def("input", &Class::getInput, py::return_value_policy::reference)
        .def("output", &Class::getOutput, py::return_value_policy::reference);
}

inline void init_detector(const py::module& m) {
    using Class = Modules::Detector<CF32, F32>;

    py::class_<Class, std::shared_ptr<Class>> detector(m, "Detector");

    py::class_<Class::Config>(detector, "Config")
        .def(py::init<const U64&,
                      const U64&,
                      const U64&,
                      const U64&,
                      const U64&,
                      const U64&,
                      const U64&>(), py::arg("number_of_beams"),
                                     py::arg("number_of_frequency_channels"),
                                     py::arg("number_of_time_samples"),
                                     py::arg("number_of_polarizations"),
                                     py::arg("integration_size"),
                                     py::arg("number_of_output_polarizations"),
                                     py::arg("block_size"));

    py::class_<Class::Input>(detector, "Input")
        .def(py::init<const Vector<Device::CUDA, CF32>&>(), py::arg("buf"));
        
    detector
        .def(py::init<const Class::Config&,
                      const Class::Input&>(), py::arg("config"),
                                              py::arg("input"))
        .def("input_size", &Class::getInputSize)
        .def("output_size", &Class::getOutputSize)
        .def("input", &Class::getInput, py::return_value_policy::reference)
        .def("output", &Class::getOutput, py::return_value_policy::reference);
}

inline void init_modules(const py::module& m) {
    init_channelizer(m);
    init_beamformer(m);
    init_phasor(m);
    init_detector(m);
}
