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
} guppiraw_block_meta_t;

void guppiraw_parse_block_meta(char* entry, void* block_meta_void) {
  guppiraw_block_meta_t* block_meta = (guppiraw_block_meta_t*) block_meta_void;
  switch (((uint64_t*)entry)[0]) {
    case KEY_UINT64_ID_LE('N','A','N','T','S',' ',' ',' '):
      hgeti4(entry, "NANTS", &block_meta->nants);
      break;
    default:
      break;
  }
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

    constexpr const U64 getNumberOfFrequencyChannels() const {
        return this->getDatashape().n_obschan;
    }

    constexpr const U64 getNumberOfOutputPolarizations() const {
        return this->getDatashape().n_pol;
    }

    constexpr const U64 getNumberOfTimeSamples() const {
        return this->getDatashape().n_time;
    }

    constexpr const U64 getOutputSize() const {
        return getNumberOfFrequencyChannels() * getNumberOfOutputPolarizations() * getNumberOfTimeSamples();
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

