#include "blade/modules/guppi/reader.hh"

#include "guppi.jit.hh"

namespace Blade::Modules::Guppi {

template<typename OT>
Reader<OT>::Reader(const Config& config, const Input& input)
        : Module(config.blockSize, guppi_kernel),
          config(config),
          input(input) {
    BL_INFO("===== GUPPI Reader Module Configuration");
    
    if (guppiraw_iterate_open_with_user_metadata(&this->gr_iterate, config.filepath.c_str(), sizeof(guppiraw_block_meta_t), guppiraw_parse_block_meta)) {
        BL_FATAL("Errored opening stem: {}.{:04d}.raw\n", this->gr_iterate.stempath, this->gr_iterate.fileenum_offset);
    }

    BL_INFO("Input File Path: {}", config.filepath);

    BL_CHECK_THROW(InitOutput(output.buf, getOutputSize()));
    
    BL_INFO("Datashape: [{}, {}, {}, {}, CI{}] ({} bytes)",
        this->getNumberOfAntenna(),
        this->getNumberOfFrequencyChannels(),
        this->getNumberOfTimeSamples(),
        this->getNumberOfPolarizations(),
        this->getDatashape()->n_bit,
        this->getDatashape()->block_size
    );

    if(this->getBlockMeta()->piperblk == 0) {
        this->getBlockMeta()->piperblk = this->getNumberOfTimeSamples();
    }

    if(this->config.step_n_aspect == 0) {
        this->config.step_n_aspect = this->getNumberOfAntenna();
    }

    if(this->config.step_n_chan == 0) {
        this->config.step_n_chan = this->getNumberOfFrequencyChannels();
    }

    if(this->config.step_n_time == 0) {
        this->config.step_n_time = this->getNumberOfTimeSamples();
    }

    this->output.buf.resize(
        this->config.step_n_aspect*
        this->config.step_n_chan*
        this->config.step_n_time*
        this->getNumberOfPolarizations()
    );
    BL_INFO("Read {} Elements, Dimension Lengths: [{}, {}, {}, {}] ({} bytes)",
        this->output.buf.size(),
        this->config.step_n_aspect,
        this->config.step_n_chan,
        this->config.step_n_time,
        this->getNumberOfPolarizations(),
        this->output.buf.size_bytes()
    );
}

template<typename OT>
Result Reader<OT>::preprocess(const cudaStream_t& stream) {

    return Result::SUCCESS;
}

template class BLADE_API Reader<CI8>;

}  // namespace Blade::Modules::Guppi
