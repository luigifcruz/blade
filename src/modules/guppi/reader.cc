#define BL_LOG_DOMAIN "M::GUPPI::READER"

#include "blade/modules/guppi/reader.hh"

#include "guppi.jit.hh"

namespace Blade::Modules::Guppi {

typedef struct {
    I32 nants;
    F64 chan_bw_mhz;
    I32 chan_start;
    F64 obs_freq_mhz;
    U64 synctime;
    U64 piperblk;
    U64 pktidx;
    F64 dut1;
} guppiraw_block_meta_t;

const U64 KEY_UINT64_SCHAN = GUPPI_RAW_KEY_UINT64_ID_LE('S','C','H','A','N',' ',' ',' ');
const U64 KEY_UINT64_CHAN_BW = GUPPI_RAW_KEY_UINT64_ID_LE('C','H','A','N','_','B','W',' ');
const U64 KEY_UINT64_OBSFREQ = GUPPI_RAW_KEY_UINT64_ID_LE('O','B','S','F','R','E','Q',' ');
const U64 KEY_UINT64_SYNCTIME = GUPPI_RAW_KEY_UINT64_ID_LE('S','Y','N','C','T','I','M','E');
const U64 KEY_UINT64_PIPERBLK = GUPPI_RAW_KEY_UINT64_ID_LE('P','I','P','E','R','B','L','K');
const U64 KEY_UINT64_PKTIDX = GUPPI_RAW_KEY_UINT64_ID_LE('P','K','T','I','D','X',' ',' ');
const U64 KEY_UINT64_DUT1 = GUPPI_RAW_KEY_UINT64_ID_LE('D','U','T','1',' ',' ',' ',' ');

void guppiraw_parse_block_meta(const char* entry, void* block_meta) {
    if        (((U64*)entry)[0] == KEY_UINT64_SCHAN) {
        hgeti4(entry, "SCHAN", &((guppiraw_block_meta_t*)block_meta)->chan_start);
    } else if (((U64*)entry)[0] == KEY_UINT64_CHAN_BW) {
        hgetr8(entry, "CHAN_BW", &((guppiraw_block_meta_t*)block_meta)->chan_bw_mhz);
    } else if (((U64*)entry)[0] == KEY_UINT64_OBSFREQ) {
        hgetr8(entry, "OBSFREQ", &((guppiraw_block_meta_t*)block_meta)->obs_freq_mhz);
    } else if (((U64*)entry)[0] == KEY_UINT64_SYNCTIME) {
        hgetu8(entry, "SYNCTIME", &((guppiraw_block_meta_t*)block_meta)->synctime);
    } else if (((U64*)entry)[0] == KEY_UINT64_PIPERBLK) {
        hgetu8(entry, "PIPERBLK", &((guppiraw_block_meta_t*)block_meta)->piperblk);
    } else if (((U64*)entry)[0] == KEY_UINT64_PKTIDX) {
        hgetu8(entry, "PKTIDX", &((guppiraw_block_meta_t*)block_meta)->pktidx);
    } else if (((U64*)entry)[0] == KEY_UINT64_DUT1) {
        hgetr8(entry, "DUT1", &((guppiraw_block_meta_t*)block_meta)->dut1);
    } 
}

inline guppiraw_block_meta_t* getBlockMeta(const guppiraw_iterate_info_t* gr_iterate_ptr) {
    return ((guppiraw_block_meta_t*) guppiraw_iterate_metadata(gr_iterate_ptr)->user_data);
}

template<typename OT>
Reader<OT>::Reader(const Config& config, const Input& input)
        : Module(config.blockSize, guppi_kernel),
          config(config),
          input(input) {
    // Check configuration.
    if (config.filepath.length() == 0) {
        BL_FATAL("Input file ({}) is invalid.", config.filepath);
        BL_CHECK_THROW(Result::ASSERTION_ERROR);
    }

    // Open GUPPI file and configure step size.
    const auto res = 
        guppiraw_iterate_open_with_user_metadata(&gr_iterate, 
                                                 config.filepath.c_str(), 
                                                 sizeof(guppiraw_block_meta_t),
                                                 guppiraw_parse_block_meta);

    if (res) {
        BL_FATAL("Errored opening stem ({}): {}.{:04d}.raw\n", res, 
                this->gr_iterate.stempath, this->gr_iterate.fileenum_offset);
        BL_CHECK_THROW(Result::ASSERTION_ERROR);
    }

    if (getBlockMeta(&gr_iterate)->piperblk == 0) {
        getBlockMeta(&gr_iterate)->piperblk = this->getDatashape()->n_time;
    }

    if (this->config.stepNumberOfAspects == 0) {
        this->config.stepNumberOfAspects = getTotalOutputBufferDims().numberOfAspects();
    }

    if (this->config.stepNumberOfFrequencyChannels == 0) {
        this->config.stepNumberOfFrequencyChannels = getTotalOutputBufferDims().numberOfFrequencyChannels();
    }

    if (this->config.stepNumberOfTimeSamples == 0) {
        this->config.stepNumberOfTimeSamples = this->getDatashape()->n_time;
    }

    // Allocate output buffers.
    BL_CHECK_THROW(output.stepDut1.resize({1}));
    BL_CHECK_THROW(output.stepJulianDate.resize({1}));
    BL_CHECK_THROW(output.stepBuffer.resize(getStepOutputBufferDims()));

    // Print configuration information.
    BL_INFO("Type: {} -> {}", "N/A", TypeInfo<OT>::name);
    BL_INFO("Step Dimensions [A, F, T, P]: {} -> {}", "N/A", getStepOutputBuffer().dims());
    BL_INFO("Total Dimensions [A, F, T, P]: {} -> {}", "N/A", getTotalOutputBufferDims());
    BL_INFO("Input File Path: {}", config.filepath);
}

template<typename OT>
const F64 Reader<OT>::getChannelBandwidth() const {
    return getBlockMeta(&gr_iterate)->chan_bw_mhz * 1e6;
}

template<typename OT>
const F64 Reader<OT>::getTotalBandwidth() const {
    return getChannelBandwidth() * getStepOutputBufferDims().numberOfFrequencyChannels();
}

template<typename OT>
const U64 Reader<OT>::getChannelStartIndex() const {
    return getBlockMeta(&gr_iterate)->chan_start;
}

template<typename OT>
const F64 Reader<OT>::getObservationFrequency() const {
    return getBlockMeta(&gr_iterate)->obs_freq_mhz * 1e6;
}

template<typename OT>
const Result Reader<OT>::preprocess(const cudaStream_t& stream) {
    if (!this->keepRunning()) {
        return Result::EXHAUSTED;
    }

    this->lastread_block_index = gr_iterate.block_index;
    this->lastread_aspect_index = gr_iterate.aspect_index;
    this->lastread_channel_index = gr_iterate.chan_index;
    this->lastread_time_index = gr_iterate.time_index;

    // Query internal library Julian Date. 
    const auto unixDate =
        guppiraw_calc_unix_date(
            1.0 / this->getChannelBandwidth(),
            this->getDatashape()->n_time,
            getBlockMeta(&gr_iterate)->piperblk,
            getBlockMeta(&gr_iterate)->synctime,
            (getBlockMeta(&gr_iterate)->pktidx + 
             (this->lastread_block_index + 
              (0.5 * getBlockMeta(&gr_iterate)->piperblk)) *
              this->getDatashape()->n_time));

    this->output.stepJulianDate[0] = calc_julian_date_from_unix(unixDate);

    // Query internal library DUT1 value.
    this->output.stepDut1[0] = getBlockMeta(&gr_iterate)->dut1;

    // Run library internal read method.
    const I64 bytes_read = 
        guppiraw_iterate_read(&this->gr_iterate,
                              this->getStepOutputBufferDims().numberOfTimeSamples(),
                              this->getStepOutputBufferDims().numberOfFrequencyChannels(),
                              this->getStepOutputBufferDims().numberOfAspects(),
                              this->output.stepBuffer.data());

    if (bytes_read <= 0) {
        BL_FATAL("File reader couldn't read bytes.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

template class BLADE_API Reader<CI8>;

}  // namespace Blade::Modules::Guppi
