#ifndef BLADE_MODULES_SETICORE_HITS_STAMP_WRITER_HH
#define BLADE_MODULES_SETICORE_HITS_STAMP_WRITER_HH

#include <filesystem>
#include <string>

#include "blade/base.hh"
#include "blade/module.hh"

extern "C" {
#include "radiointerferometryc99.h"
}

#include "dedoppler.h"
#include "hit_file_writer.h"
#include "stamp_extractor.h"
#include "dedoppler_hit_group.h"

#include <capnp/message.h>
#include <capnp/serialize-packed.h>
#include "stamp.capnp.h"


namespace Blade::Modules::Seticore {

template<typename IT>
class BLADE_API HitsStampWriter : public Module {
 public:
    // Configuration 

    struct Config {
        std::string filepathPrefix;

        U64 telescopeId;
        std::string sourceName;
        std::string observationIdentifier;
        RA_DEC phaseCenter;
        U64 coarseStartChannelIndex;
        U64 coarseChannelRatio;
        F64 channelBandwidthHz;
        F64 channelTimespanS;
        F64 stampFrequencyMarginHz = 500.0;
        I64 hitsGroupingMargin = 30;

        U64 blockSize = 512;
    };

    constexpr const Config& getConfig() const {
        return this->config;
    }

    // Input

    struct Input {
        const ArrayTensor<Device::CPU, IT>& buffer;
        std::vector<DedopplerHit>& hits;
        const Vector<Device::CPU, F64>& frequencyOfFirstChannelHz;
        const Vector<Device::CPU, F64>& julianDateStart;
    };

    constexpr const ArrayTensor<Device::CPU, IT>& getInputBuffer() const {
        return this->input.buffer;
    }

    constexpr const Vector<Device::CPU, F64>& getInputFrequencyOfFirstChannelHz() const {
        return this->input.frequencyOfFirstChannelHz;
    }

    constexpr const Vector<Device::CPU, F64>& getInputJulianDateStart() const {
        return this->input.julianDateStart;
    }

    // Output

    struct Output {
    };

    // Constructor & Processing

    explicit HitsStampWriter(const Config& config, const Input& input);
    ~HitsStampWriter();
    
    const Result process(const cudaStream_t& stream = 0) final;

    // Miscellaneous

 private:
    // Variables

    Config config;
    Input input;
    Output output;
    
    U64 stampsWritten;
    U64 fileId;
    I32 fileDescriptor;
};

}  // namespace Blade::Modules::Seticore

#endif
