#ifndef BLADE_MODULES_CHANNELIZER_HH
#define BLADE_MODULES_CHANNELIZER_HH

#include <string>
#include <cufft.h>

#include "blade/base.hh"
#include "blade/module.hh"

namespace Blade::Modules {

template<typename IT, typename OT>
class BLADE_API Channelizer : public Module {
 public:
    struct Config {
        U64 numberOfBeams;
        U64 numberOfAntennas;
        U64 numberOfFrequencyChannels;
        U64 numberOfTimeSamples;
        U64 numberOfPolarizations; 

        U64 rate = 4;
        U64 blockSize = 512;
    };

    struct Input {
        const Vector<Device::CUDA, IT>& buf;
    };

    struct Output {
        Vector<Device::CUDA, OT> buf;
    };

    explicit Channelizer(const Config& config, const Input& input);

    constexpr Vector<Device::CUDA, IT>& getInput() {
        return const_cast<Vector<Device::CUDA, IT>&>(this->input.buf);
    }

    constexpr const Vector<Device::CUDA, OT>& getOutput() const {
        return this->output.buf;
    }

    constexpr const Config& getConfig() const {
        return this->config;
    }

    constexpr const U64 getBufferSize() const {
        return config.numberOfPolarizations * 
               config.numberOfTimeSamples *
               config.numberOfAntennas * 
               config.numberOfFrequencyChannels;
    }

    Result process(const cudaStream_t& stream = 0) final;

 private:
    const Config config;
    const Input input;
    Output output;

    cufftHandle plan;

    Result initializeCufft();
    Result initializeInternal();
};

}  // namespace Blade::Modules

#endif
