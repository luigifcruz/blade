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
        BOOL searchIncoherentBeam = true;

        std::string filepathPrefix;
        U64 telescopeId;
        std::string sourceName;
        std::string observationIdentifier;
        RA_DEC phaseCenter;
        U64 coarseStartChannelIndex;
        F64 julianDateStart;
        std::vector<std::string> aspectNames;
        std::vector<RA_DEC> aspectCoordinates;
        U64 totalNumberOfTimeSamples;
        U64 totalNumberOfFrequencyChannels;

        U64 blockSize = 512;
    };

    constexpr const Config& getConfig() const {
        return config;
    }

    // Input

    void setFrequencyOfFirstInputChannel(F64 hz);

    struct Input {
        const ArrayTensor<Device::CUDA, F32>& buf;
        const Vector<Device::CPU, U64>& coarseFrequencyChannelOffset;
    };

    constexpr const Vector<Device::CPU, U64>& getInputCoarseFrequencyChannelOffset() {
        return this->input.coarseFrequencyChannelOffset;
    }

    // Output

    struct Output {
        std::vector<DedopplerHit> hits;
    };

    constexpr std::vector<DedopplerHit>& getOutputHits() {
        return this->output.hits;
    }

    // Constructor & Processing

    explicit Dedoppler(const Config& config, const Input& input);
    const Result process(const cudaStream_t& stream = 0) final;

    private:
    // Variables

    ArrayTensor<Device::CPU, F32> buf;

    const Config config;
    const Input input;
    Output output;

    Dedopplerer dedopplerer;
    FilterbankMetadata metadata;
    unique_ptr<HitFileWriter> hit_recorder;
};

} // namespace Blade::Modules::Seticore

#endif
