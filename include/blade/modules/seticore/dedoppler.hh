#ifndef BLADE_MODULES_SETICORE_DEDOPPLER_HH
#define BLADE_MODULES_SETICORE_DEDOPPLER_HH

#include "blade/base.hh"
#include "blade/module.hh"

#include "dedoppler.h"
#include "filterbank_buffer.h"
#include "hit_file_writer.h"
#include "hit_recorder.h"

namespace Blade::Modules::Seticore {

class BLADE_API Dedoppler : public Module {
    public:
    // Configuration

    struct Config {
        BOOL mitigateDcSpike;
        F64 minimumDriftRate = 0.0;
        F64 maximumDriftRate;
        F64 snrThreshold;

        F64 channelBandwidthHz;
        F64 channelTimespanS;
        U64 coarseChannelRate;
        BOOL lastBeamIsIncoherent = false;

        U64 blockSize = 512;
    };

    constexpr const Config& getConfig() const {
        return config;
    }

    // Input

    void setFrequencyOfFirstInputChannel(F64 hz);

    struct Input {
        const ArrayTensor<Device::CUDA, F32>& buf;
    };

    // Output

    struct Output {
        std::vector<DedopplerHit> hits;
    };

    constexpr const std::vector<DedopplerHit>& getOutputHits() const {
        return this->output.hits;
    }

    // Constructor & Processing

    explicit Dedoppler(const Config& config, const Input& input);
    const Result process(const cudaStream_t& stream = 0) final;

    private:
    // Variables

    const Config config;
    const Input input;
    Output output;

    Dedopplerer dedopplerer;
    FilterbankMetadata metadata;
};

} // namespace Blade::Modules::Seticore

#endif
