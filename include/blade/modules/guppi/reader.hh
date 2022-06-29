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

    constexpr const U64 getNumberOfFrequencyChannels() const {
        return this->gr_blockinfo.n_obschan;
    }

    constexpr const U64 getNumberOfOutputPolarizations() const {
        return this->gr_blockinfo.n_pol;
    }

    constexpr const U64 getNumberOfTimeSamples() const {
        return this->gr_blockinfo.n_time;
    }

    constexpr const U64 getOutputSize() const {
        return gr_blockinfo.n_obschan * gr_blockinfo.n_time * gr_blockinfo.n_pol;
    }

    Result preprocess(const cudaStream_t& stream = 0) final;

 private:
    const Config config;
    const Input input;
    Output output;

    int input_fd;
    guppiraw_block_info_t gr_blockinfo;
};

}  // namespace Blade::Modules

#endif

