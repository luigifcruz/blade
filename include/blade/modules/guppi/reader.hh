#ifndef BLADE_MODULES_GUPPI_READER_HH
#define BLADE_MODULES_GUPPI_READER_HH

#include <filesystem>
#include <string>

#include "blade/base.hh"
#include "blade/module.hh"

extern "C" {
#include "guppiraw.h"
}

namespace Blade::Modules::Guppi {

typedef struct {
  int nants;
  double chan_bw_mhz;
  int chan_start;
  double obs_freq_mhz;
} guppiraw_block_meta_t;

const uint64_t KEY_NANTS_UINT64 = GUPPI_RAW_KEY_UINT64_ID_LE('N','A','N','T','S',' ',' ',' ');
const uint64_t KEY_SCHAN_UINT64 = GUPPI_RAW_KEY_UINT64_ID_LE('S','C','H','A','N',' ',' ',' ');
const uint64_t KEY_CHAN_BW_UINT64 = GUPPI_RAW_KEY_UINT64_ID_LE('C','H','A','N','_','B','W',' ');
const uint64_t KEY_OBSFREQ_UINT64 = GUPPI_RAW_KEY_UINT64_ID_LE('O','B','S','F','R','E','Q',' ');

void guppiraw_parse_block_meta(char* entry, void* block_meta_void) {
  guppiraw_block_meta_t* block_meta = (guppiraw_block_meta_t*) block_meta_void;
  if(((uint64_t*)entry)[0] == KEY_NANTS_UINT64)
    hgeti4(entry, "NANTS", &block_meta->nants);
  else if(((uint64_t*)entry)[0] == KEY_SCHAN_UINT64)
    hgeti4(entry, "SCHAN", &block_meta->chan_start);
  else if(((uint64_t*)entry)[0] == KEY_CHAN_BW_UINT64)
    hgetr8(entry, "CHAN_BW", &block_meta->chan_bw_mhz);
  else if(((uint64_t*)entry)[0] == KEY_OBSFREQ_UINT64)
    hgetr8(entry, "OBSFREQ", &block_meta->obs_freq_mhz);
}

template<typename OT>
class BLADE_API Reader : public Module {
 public:
    struct Config {
        std::string filepath;

        U64 blockSize = 512;
    };

    struct Input {
    };

    struct Output {
        Vector<Device::CPU, OT> buf;
    };

    explicit Reader(const Config& config, const Input& input);

    constexpr const Vector<Device::CPU, OT>& getOutput() const {
        return this->output.buf;
    }

    constexpr const Config& getConfig() const {
        return this->config;
    }

    constexpr const U64 getNumberOfAntenna() const {
        return ((guppiraw_block_meta_t*)this->gr_iterate.file_info.block_info.header_user_data)->nants;
    }

    constexpr const F64 getBandwidthOfChannel() const {
        return ((guppiraw_block_meta_t*)this->gr_iterate.file_info.block_info.header_user_data)->chan_bw_mhz * 1e6;
    }

    constexpr const U64 getChannelStartIndex() const {
        return ((guppiraw_block_meta_t*)this->gr_iterate.file_info.block_info.header_user_data)->chan_start;
    }

    constexpr const F64 getBandwidthCenter() const {
        return ((guppiraw_block_meta_t*)this->gr_iterate.file_info.block_info.header_user_data)->obs_freq_mhz * 1e6;
    }

    constexpr const U64 getNumberOfFrequencyChannels() const {
        return this->getDatashape().n_obschan;
    }

    constexpr const U64 getNumberOfPolarizations() const {
        return this->getDatashape().n_pol;
    }

    constexpr const U64 getNumberOfTimeSamples() const {
        return this->getDatashape().n_time;
    }

    constexpr const U64 getOutputSize() const {
        return getNumberOfFrequencyChannels() * getNumberOfPolarizations() * getNumberOfTimeSamples();
    }

    Result preprocess(const cudaStream_t& stream = 0) final;

 private:
    const Config config;
    const Input input;
    Output output;

    guppiraw_iterate_info_t gr_iterate;

    constexpr const guppiraw_datashape_t getDatashape() const {
        return this->gr_iterate.file_info.block_info.datashape;
    }
};

}  // namespace Blade::Modules

#endif

