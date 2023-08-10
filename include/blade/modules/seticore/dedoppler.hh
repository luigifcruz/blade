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

        F64 frequencyOfFirstChannelHz;
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
        std::vector<std::string> aspectNames;
        std::vector<RA_DEC> aspectCoordinates;
        U64 totalNumberOfTimeSamples;
        U64 totalNumberOfFrequencyChannels;

        BOOL produceDebugHits = false;

        U64 blockSize = 512;
    };

    constexpr const Config& getConfig() const {
        return config;
    }

    // Input

    struct Input {
        const ArrayTensor<Device::CPU, F32>& buf;
        const Vector<Device::CPU, U64>& coarseFrequencyChannelOffset;
        const Vector<Device::CPU, F64>& julianDate;
    };

    constexpr const Vector<Device::CPU, U64>& getInputCoarseFrequencyChannelOffset() {
        return this->input.coarseFrequencyChannelOffset;
    }

    constexpr const Vector<Device::CPU, F64>& getJulianDateOfInput() {
        return this->input.julianDate;
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

    ArrayTensor<Device::CUDA, F32> searchBuffer;
    ArrayTensor<Device::CUDA, F32> incohBuffer;

    const Config config;
    const Input input;
    Output output;

    Dedopplerer dedopplerer;
    FilterbankMetadata metadata;
    unique_ptr<HitFileWriter> hit_recorder;
};

} // namespace Blade::Modules::Seticore

#endif
