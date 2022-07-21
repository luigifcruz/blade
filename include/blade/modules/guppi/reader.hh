#ifndef BLADE_MODULES_GUPPI_READER_HH
#define BLADE_MODULES_GUPPI_READER_HH

#include <filesystem>
#include <string>

#include "blade/base.hh"
#include "blade/module.hh"

extern "C" {
#include "guppiraw.h"
#include "radiointerferometryc99.h"
}

namespace Blade::Modules::Guppi {

template<typename OT>
class BLADE_API Reader : public Module {
 public:
    struct Config {
        std::string filepath;

        U64 stepNumberOfTimeSamples;
        U64 stepNumberOfFrequencyChannels;
        U64 stepNumberOfAntennas;

        U64 blockSize = 512;
    };

    struct Input {
    };

    struct Output {
        Vector<Device::CPU, OT> buf;
    };

    explicit Reader(const Config& config, const Input& input);

    constexpr const bool canRead() const {
        return !this->flag_error &&
            guppiraw_iterate_ntime_remaining(&this->gr_iterate) > this->getNumberOfTimeSamples();
    }

    constexpr const Vector<Device::CPU, OT>& getOutput() {
        this->lastread_block_index++;
        this->lastread_aspect_index = gr_iterate.aspect_index;
        this->lastread_channel_index = gr_iterate.chan_index;
        this->lastread_time_index = gr_iterate.time_index;

        const I64 bytes_read = guppiraw_iterate_read(
            &this->gr_iterate,
            this->getNumberOfTimeSamples(),
            this->getNumberOfFrequencyChannels(),
            this->getNumberOfAntennas(),
            this->output.buf.data()
        );

        if (bytes_read <= 0) {
            BL_ERROR("Guppi::Reader encountered error: {}.", bytes_read);
            this->flag_error = true; // << HAS TO GO 
        }

        return this->output.buf;
    }

    constexpr const Config& getConfig() const {
        return this->config;
    }

    const F64 getBandwidthOfChannel();
    const U64 getChannelStartIndex();
    const F64 getBandwidthCenter();

    constexpr const U64 getTotalNumberOfAntennas() const {
        return this->getDatashape()->n_aspect;
    }

    constexpr const U64 getTotalNumberOfFrequencyChannels() const {
        return this->getDatashape()->n_aspectchan;
    }

    constexpr const U64 getTotalNumberOfPolarizations() const {
        return this->getDatashape()->n_pol;
    }

    constexpr const U64 getTotalNumberOfTimeSamples() const {
        return this->getDatashape()->n_time;
    }

    constexpr const U64 getTotalOutputSize() const {
        return this->getTotalNumberOfAntennas() *
               this->getTotalNumberOfFrequencyChannels() *
               this->getTotalNumberOfTimeSamples() * 
               this->getTotalNumberOfPolarizations();
    }

    constexpr const U64 getNumberOfAntennas() const {
        return this->config.stepNumberOfAntennas;
    }

    constexpr const U64 getNumberOfFrequencyChannels() const {
        return this->config.stepNumberOfFrequencyChannels;
    }

    constexpr const U64 getNumberOfPolarizations() const {
        return this->getTotalNumberOfPolarizations();
    }

    constexpr const U64 getNumberOfTimeSamples() const {
        return this->config.stepNumberOfTimeSamples;
    }

    constexpr const U64 getOutputSize() const {
        return this->getNumberOfAntennas() *
               this->getNumberOfFrequencyChannels() *
               this->getNumberOfTimeSamples() * 
               this->getNumberOfPolarizations();
    }

    constexpr const F64 getBlockEpochSeconds() {
        return this->getBlockEpochSeconds(0);
    }
    const F64 getBlockEpochSeconds(U64 block_time_offset);

    Result preprocess(const cudaStream_t& stream = 0) final;

 private:
    Config config;
    const Input input;
    Output output;

    bool flag_error = false; // << HAS TO GO

    I64 lastread_block_index = -1;
    U64 lastread_aspect_index;
    U64 lastread_channel_index;
    U64 lastread_time_index;

    guppiraw_iterate_info_t gr_iterate = {0};

    constexpr const guppiraw_datashape_t* getDatashape() const {
        return guppiraw_iterate_datashape(&this->gr_iterate);
    }
};

}  // namespace Blade::Modules

#endif
