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

typedef struct {
  int nants;
  double chan_bw_mhz;
  int chan_start;
  double obs_freq_mhz;
  uint64_t synctime;
  uint64_t piperblk;
  uint64_t pktidx;
} guppiraw_block_meta_t;

const uint64_t KEY_SCHAN_UINT64 = GUPPI_RAW_KEY_UINT64_ID_LE('S','C','H','A','N',' ',' ',' ');
const uint64_t KEY_CHAN_BW_UINT64 = GUPPI_RAW_KEY_UINT64_ID_LE('C','H','A','N','_','B','W',' ');
const uint64_t KEY_OBSFREQ_UINT64 = GUPPI_RAW_KEY_UINT64_ID_LE('O','B','S','F','R','E','Q',' ');
const uint64_t KEY_SYNCTIME_UINT64 = GUPPI_RAW_KEY_UINT64_ID_LE('S','Y','N','C','T','I','M','E');
const uint64_t KEY_PIPERBLK_UINT64 = GUPPI_RAW_KEY_UINT64_ID_LE('P','I','P','E','R','B','L','K');
const uint64_t KEY_PKTIDX_UINT64 = GUPPI_RAW_KEY_UINT64_ID_LE('P','K','T','I','D','X',' ',' ');

void guppiraw_parse_block_meta(const char* entry, void* block_meta) {
  if(((uint64_t*)entry)[0] == KEY_SCHAN_UINT64)
    hgeti4(entry, "SCHAN", &((guppiraw_block_meta_t*)block_meta)->chan_start);
  else if(((uint64_t*)entry)[0] == KEY_CHAN_BW_UINT64)
    hgetr8(entry, "CHAN_BW", &((guppiraw_block_meta_t*)block_meta)->chan_bw_mhz);
  else if(((uint64_t*)entry)[0] == KEY_OBSFREQ_UINT64)
    hgetr8(entry, "OBSFREQ", &((guppiraw_block_meta_t*)block_meta)->obs_freq_mhz);
  else if(((uint64_t*)entry)[0] == KEY_SYNCTIME_UINT64)
    hgetu8(entry, "SYNCTIME", &((guppiraw_block_meta_t*)block_meta)->synctime);
  else if(((uint64_t*)entry)[0] == KEY_PIPERBLK_UINT64)
    hgetu8(entry, "PIPERBLK", &((guppiraw_block_meta_t*)block_meta)->piperblk);
  else if(((uint64_t*)entry)[0] == KEY_PKTIDX_UINT64)
    hgetu8(entry, "PKTIDX", &((guppiraw_block_meta_t*)block_meta)->pktidx);
}

template<typename OT>
class BLADE_API Reader : public Module {
 public:
    struct Config {
        std::string filepath;

        U64 step_n_time;
        U64 step_n_chan;
        U64 step_n_aspect;

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
            guppiraw_iterate_ntime_remaining(&this->gr_iterate) > this->config.step_n_time
        ;
    }

    constexpr const Vector<Device::CPU, OT>& getOutput() {
        this->lastread_block_index++;
        this->lastread_aspect_index = gr_iterate.aspect_index;
        this->lastread_channel_index = gr_iterate.chan_index;
        this->lastread_time_index = gr_iterate.time_index;
        const I64 bytes_read = guppiraw_iterate_read(
            &this->gr_iterate,
            this->config.step_n_time,
            this->config.step_n_chan,
            this->config.step_n_aspect,
            this->output.buf.data()
        );
        if(bytes_read <= 0) {
            BL_ERROR("Guppi::Reader encountered error: {}.", bytes_read);
            this->flag_error = true;
        }
        return this->output.buf;
    }

    constexpr const Config& getConfig() const {
        return this->config;
    }

    constexpr const F64 getBandwidthOfChannel() const {
        return this->getBlockMeta()->chan_bw_mhz * 1e6;
    }

    constexpr const U64 getChannelStartIndex() const {
        return this->getBlockMeta()->chan_start;
    }

    constexpr const F64 getBandwidthCenter() const {
        return this->getBlockMeta()->obs_freq_mhz * 1e6;
    }

    constexpr const U64 getNumberOfAntenna() const {
        return this->getDatashape()->n_aspect;
    }

    constexpr const U64 getNumberOfFrequencyChannels() const {
        return this->getDatashape()->n_aspectchan;
    }

    constexpr const U64 getNumberOfPolarizations() const {
        return this->getDatashape()->n_pol;
    }

    constexpr const U64 getNumberOfTimeSamples() const {
        return this->getDatashape()->n_time;
    }

    constexpr const U64 getOutputSize() const {
        return getNumberOfFrequencyChannels() * getNumberOfPolarizations() * getNumberOfTimeSamples();
    }

    constexpr const F64 getBlockEpochSeconds() {
        return this->getBlockEpochSeconds(0);
    }
    constexpr const F64 getBlockEpochSeconds(U64 block_time_offset) {
        return calc_epoch_seconds_from_guppi_param(
            1.0/this->getBandwidthOfChannel(),
            this->getNumberOfTimeSamples(),
            this->getBlockMeta()->piperblk,
            this->getBlockMeta()->synctime,
            this->getBlockMeta()->pktidx + 
                (this->lastread_block_index + block_time_offset)*this->getNumberOfTimeSamples()
        );
    }

    Result preprocess(const cudaStream_t& stream = 0) final;

 private:
    Config config;
    const Input input;
    Output output;
    bool flag_error = false;

    I64 lastread_block_index = -1;
    U64 lastread_aspect_index;
    U64 lastread_channel_index;
    U64 lastread_time_index;

    guppiraw_iterate_info_t gr_iterate = {0};

    constexpr const guppiraw_datashape_t* getDatashape() const {
        return guppiraw_iterate_datashape(&this->gr_iterate);
    }

    constexpr guppiraw_block_meta_t* getBlockMeta() const {
        return ((guppiraw_block_meta_t*) guppiraw_iterate_metadata(&this->gr_iterate));
    }
};

}  // namespace Blade::Modules

#endif
