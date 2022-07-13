#include "blade/modules/guppi/reader.hh"

#include "guppi.jit.hh"

namespace Blade::Modules::Guppi {

template<typename OT>
Reader<OT>::Reader(const Config& config, const Input& input)
        : Module(config.blockSize, guppi_kernel),
          config(config),
          input(input) {
    BL_INFO("===== GUPPI Reader Module Configuration");

    this->gr_iterate.file_info.block_info.metadata.user_data = calloc(sizeof(guppiraw_block_meta_t), 1);
    this->gr_iterate.file_info.block_info.metadata.user_callback = guppiraw_parse_block_meta;
    
    if (guppiraw_iterate_open_stem(config.filepath.c_str(), &this->gr_iterate)) {
        BL_FATAL("Could not open: {}.{:04d}.raw\n", this->gr_iterate.stempath, this->gr_iterate.fileenum);
    }

    BL_INFO("Input File Path: {}", config.filepath);

    BL_CHECK_THROW(InitOutput(output.buf, getOutputSize()));
    
    BL_INFO("Datashape: [{}, {}, {}, {}, CI{}] ({} bytes)",
        this->getNumberOfAntenna(),
        this->getNumberOfFrequencyChannels(),
        this->getNumberOfTimeSamples(),
        this->getNumberOfPolarizations(),
        this->getDatashape().n_bit,
        this->getDatashape().block_size
    );

    if(this->getBlockMeta()->piperblk == 0) {
        this->getBlockMeta()->piperblk = this->getNumberOfTimeSamples();
    }
    this->block_pktidx = this->getBlockMeta()->pktidx;

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
