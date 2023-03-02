#ifndef BLADE_MODULES_FILTERBANK_WRITER_HH
#define BLADE_MODULES_FILTERBANK_WRITER_HH

#include <filesystem>
#include <string>
#include <fcntl.h>
#include <sys/uio.h>
#include <limits.h>

#include "blade/base.hh"
#include "blade/module.hh"

extern "C" {
#include "filterbankc99.h"
}

namespace Blade::Modules::Filterbank {

template<typename InputType>
class BLADE_API Writer : public Module {
 public:
    // Configuration

    struct Config {
        std::string filepath;

        int machineId;
        std::string telescopeName;
        int baryCentric;
        int pulsarCentric;
        double azimuthStart;
        double zenithStart;
        double firstChannelFrequencyHz;
        double bandwidthHz;
        double julianDateStart;
        int numberOfIfChannels;
        std::string sourceDataFilename;
        std::vector<std::string> beamNames;
        std::vector<RA_DEC> beamCoordinates;

        U64 numberOfInputFrequencyChannelBatches = 1;
        U64 blockSize = 512;
    };

    // Input

    struct Input {
        const ArrayTensor<Device::CPU, InputType>& buffer;
    };

    constexpr const ArrayTensor<Device::CPU, InputType>& getInputBuffer() const {
        return this->input.buffer;
    }

    // Output

    struct Output {
    };

    // Constructor & Processing

    explicit Writer(const Config& config, const Input& input);
    const Result preprocess(const cudaStream_t& stream, const U64& currentComputeCount) final;

 private:
    // Miscellaneous

    void openFilesWriteHeaders();

    // Variables

    Config config;
    Input input;
    Output output;

    std::vector<I32> fileDescriptors;

    filterbank_header_t filterbank_header = {0};
};

}  // namespace Blade::Modules

#endif

