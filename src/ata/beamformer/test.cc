#include "bl-beamformer/ata/beamformer/test.hh"

namespace py = pybind11;

namespace BL::ATA::Beamformer {

Test::Test() {
    py::scoped_interpreter python;
    auto math = py::module::import("pi");
    std::cout << math.attr("sqrt")(2.0).cast<double>() << std::endl;
}

Test::~Test() {
}

} // namespace BL::ATA::Beamformer
