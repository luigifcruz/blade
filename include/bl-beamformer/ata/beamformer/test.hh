#ifndef BL_ATA_PHASORS_H
#define BL_ATA_PHASORS_H

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

#include "bl-beamformer/helpers.hh"
#include "bl-beamformer/type.hh"

namespace BL::ATA::Beamformer {

class BL_API Test {
public:
    Test();
    ~Test();

    Result beamform();

    std::span<const std::complex<int8_t>> getInputData();
    std::span<const std::complex<float>> getPhasorsData();
    std::span<const std::complex<float>> getOutputData();

private:
    py::scoped_interpreter guard{}; // WARNING: Interpreter should be destructed last!
    py::object lib;

    template<typename IT, typename OT>
    static std::span<OT> __convert(const py::object & input);
};

} // namespace BL::ATA::Beamformer

#endif

