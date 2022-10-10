#ifndef BLADE_MODULES_SETICORE_DEDOPPLER_HH
#define BLADE_MODULES_SETICORE_DEDOPPLER_HH

#include "blade/base.hh"
#include "blade/module.hh"

#include "dedoppler.h"
#include "filterbank_buffer.h"

namespace Blade::Modules::Seticore {

class BLADE_API Dedoppler : public Module {
    public:
    // Configuration

    struct Config {
        BOOL mitigateDcSpike;
        F64 observationBandwidthHz;
        F64 minimumDriftrate = 0.0;
        F64 maximumDriftrate;
        F64 snrThreshold;

        U64 blockSize = 512;
    };

    constexpr const Config& getConfig() const {
        return config;
    }

    // Input 

    struct Input {
        const ArrayTensor<Device::CPU, F32>& buf;
    };

    // Output

    struct Output {
    };

    // Constructor & Processing

    explicit Dedoppler(const Config& config, const Input& input);
    const Result process(const cudaStream_t& stream = 0) final;

    private:
    // Variables 

    const Config config;
    const Input input;
    Output output;

    Dedopplerer dedopplerer;
};

} // namespace Blade::Modules::Seticore

#endif
