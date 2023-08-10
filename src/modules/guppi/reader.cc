#define BL_LOG_DOMAIN "M::GUPPI::READER"

#include "blade/modules/guppi/reader.hh"

#include "guppi.jit.hh"

namespace Blade::Modules::Guppi {

typedef struct {
    I32 nants;
    I32 fenchan;
    F64 chan_bw_mhz;
    F64 chan_timespan;
    I32 chan_start;
    F64 obs_freq_mhz;
    U64 synctime;
    U64 piperblk;
    U64 pktidx;
    F64 dut1;
    F64 az;
    F64 el;
    F64 ra_hour;
    F64 dec_deg;
    F64 ra_phase_hour;
    F64 dec_phase_deg;
    char src_name[72];
    char telescope_id[72];
} guppiraw_block_meta_t;

const U64 KEY_UINT64_SCHAN = GUPPI_RAW_KEY_UINT64_ID_LE('S','C','H','A','N',' ',' ',' ');
const U64 KEY_UINT64_FENCHAN = GUPPI_RAW_KEY_UINT64_ID_LE('F','E','N','C','H','A','N',' ');
const U64 KEY_UINT64_CHAN_BW = GUPPI_RAW_KEY_UINT64_ID_LE('C','H','A','N','_','B','W',' ');
const U64 KEY_UINT64_TBIN = GUPPI_RAW_KEY_UINT64_ID_LE('T','B','I','N',' ',' ',' ',' ');
const U64 KEY_UINT64_OBSFREQ = GUPPI_RAW_KEY_UINT64_ID_LE('O','B','S','F','R','E','Q',' ');
const U64 KEY_UINT64_SYNCTIME = GUPPI_RAW_KEY_UINT64_ID_LE('S','Y','N','C','T','I','M','E');
const U64 KEY_UINT64_PIPERBLK = GUPPI_RAW_KEY_UINT64_ID_LE('P','I','P','E','R','B','L','K');
const U64 KEY_UINT64_PKTIDX = GUPPI_RAW_KEY_UINT64_ID_LE('P','K','T','I','D','X',' ',' ');
const U64 KEY_UINT64_DUT1 = GUPPI_RAW_KEY_UINT64_ID_LE('D','U','T','1',' ',' ',' ',' ');
const U64 KEY_UINT64_AZ = GUPPI_RAW_KEY_UINT64_ID_LE('A','Z',' ',' ',' ',' ',' ',' ');
const U64 KEY_UINT64_EL = GUPPI_RAW_KEY_UINT64_ID_LE('E','L',' ',' ',' ',' ',' ',' ');
const U64 KEY_UINT64_RA = GUPPI_RAW_KEY_UINT64_ID_LE('R','A','_','S','T','R',' ',' ');
const U64 KEY_UINT64_DEC = GUPPI_RAW_KEY_UINT64_ID_LE('D','E','C','_','S','T','R',' ');
const U64 KEY_UINT64_RA_PHASE = GUPPI_RAW_KEY_UINT64_ID_LE('R','A','_','P','H','A','S',' ');
const U64 KEY_UINT64_DEC_PHASE = GUPPI_RAW_KEY_UINT64_ID_LE('D','E','C','_','P','H','A','S');
const U64 KEY_UINT64_SRC_NAME = GUPPI_RAW_KEY_UINT64_ID_LE('S','R','C','_','N','A','M','E');
const U64 KEY_UINT64_TELESCOP = GUPPI_RAW_KEY_UINT64_ID_LE('T','E','L','E','S','C','O','P');

void guppiraw_parse_block_meta(const char* entry, void* block_meta) {
    if        (((U64*)entry)[0] == KEY_UINT64_SCHAN) {
        hgeti4(entry, "SCHAN", &((guppiraw_block_meta_t*)block_meta)->chan_start);
    } else if (((U64*)entry)[0] == KEY_UINT64_FENCHAN) {
        hgeti4(entry, "FENCHAN", &((guppiraw_block_meta_t*)block_meta)->fenchan);
    } else if (((U64*)entry)[0] == KEY_UINT64_CHAN_BW) {
        hgetr8(entry, "CHAN_BW", &((guppiraw_block_meta_t*)block_meta)->chan_bw_mhz);
    } else if (((U64*)entry)[0] == KEY_UINT64_TBIN) {
        hgetr8(entry, "TBIN", &((guppiraw_block_meta_t*)block_meta)->chan_timespan);
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
    } else if (((U64*)entry)[0] == KEY_UINT64_AZ) {
        hgetr8(entry, "AZ", &((guppiraw_block_meta_t*)block_meta)->az);
    } else if (((U64*)entry)[0] == KEY_UINT64_EL) {
        hgetr8(entry, "EL", &((guppiraw_block_meta_t*)block_meta)->el);
    } else if (((U64*)entry)[0] == KEY_UINT64_RA) {
        hgetr8(entry, "RA_STR", &((guppiraw_block_meta_t*)block_meta)->ra_hour);
    } else if (((U64*)entry)[0] == KEY_UINT64_DEC) {
        hgetr8(entry, "DEC_STR", &((guppiraw_block_meta_t*)block_meta)->dec_deg);
    } else if (((U64*)entry)[0] == KEY_UINT64_RA_PHASE) {
        hgetr8(entry, "RA_PHASE", &((guppiraw_block_meta_t*)block_meta)->ra_phase_hour);
    } else if (((U64*)entry)[0] == KEY_UINT64_DEC_PHASE) {
        hgetr8(entry, "DEC_PHASE", &((guppiraw_block_meta_t*)block_meta)->dec_phase_deg);
    } else if (((U64*)entry)[0] == KEY_UINT64_SRC_NAME) {
        hgets(entry, "SRC_NAME", 72, ((guppiraw_block_meta_t*)block_meta)->src_name);
    } else if (((U64*)entry)[0] == KEY_UINT64_TELESCOP) {
        hgets(entry, "TELESCOP", 72, ((guppiraw_block_meta_t*)block_meta)->telescope_id);
    } else if (guppiraw_header_entry_is_END((U64*)entry)) {
        if (((guppiraw_block_meta_t*)block_meta)->src_name[0] == '\0') {
            strcpy(((guppiraw_block_meta_t*)block_meta)->src_name, "Unknown");
        }
        if (((guppiraw_block_meta_t*)block_meta)->telescope_id[0] == '\0') {
            strcpy(((guppiraw_block_meta_t*)block_meta)->telescope_id, "Unknown");
        }
        if (((guppiraw_block_meta_t*)block_meta)->chan_timespan == 0) {
            ((guppiraw_block_meta_t*)block_meta)->chan_timespan = 1e-6 / abs(((guppiraw_block_meta_t*)block_meta)->chan_bw_mhz);
        }
    }
}

inline guppiraw_block_meta_t* getBlockMeta(const guppiraw_iterate_info_t* gr_iterate_ptr) {
    return ((guppiraw_block_meta_t*) guppiraw_iterate_metadata(gr_iterate_ptr)->user_data);
}

template<typename OT>
Reader<OT>::Reader(const Config& config, const Input& input)
        : Module(guppi_program),
          config(config),
          input(input) {
    // Check configuration.
    if (config.filepath.length() == 0) {
        BL_FATAL("Input file ({}) is invalid.", config.filepath);
        BL_CHECK_THROW(Result::ASSERTION_ERROR);
    }

    this->gr_iterate.n_file = config.numberOfFilesLimit;
    // Open GUPPI file and configure step size.
    const auto res =
        guppiraw_iterate_open_with_user_metadata(&gr_iterate,
                                                 config.filepath.c_str(),
                                                 sizeof(guppiraw_block_meta_t),
                                                 guppiraw_parse_block_meta);

    if (res) {
        BL_FATAL("Errored opening stem ({}): {}.{:04d}.raw", res,
                this->gr_iterate.stempath, this->gr_iterate.fileenum_offset);
        BL_DEBUG("\tin file {}", this->gr_iterate.fileenum_offset + this->gr_iterate.n_file);
        guppiraw_file_info_t* gr_fileinfo = guppiraw_iterate_file_info(&this->gr_iterate, this->gr_iterate.n_file-1);
        BL_DEBUG("\taround block #{}/{}", gr_fileinfo->block_index, gr_fileinfo->n_block);
        BL_DEBUG("\tblock #{} @ {}/{}", gr_fileinfo->block_index, gr_fileinfo->file_header_pos[gr_fileinfo->block_index], gr_fileinfo->bytesize_file);
        BL_DEBUG("\tblock #{} @ {}/{}", gr_fileinfo->block_index-1, gr_fileinfo->file_header_pos[gr_fileinfo->block_index-1], gr_fileinfo->bytesize_file);
        BL_CHECK_THROW(Result::ASSERTION_ERROR);
    }

    if (getBlockMeta(&gr_iterate)->piperblk == 0) {
        getBlockMeta(&gr_iterate)->piperblk = this->getDatashape()->n_time;
    }

    if (this->config.stepNumberOfFrequencyChannels == 0) {
        this->config.stepNumberOfFrequencyChannels = getTotalOutputBufferDims().numberOfFrequencyChannels();
    }

    if (this->config.stepNumberOfTimeSamples == 0) {
        this->config.stepNumberOfTimeSamples = this->getDatashape()->n_time;
    }

    // Print configuration information.
    const auto totalDims = getTotalOutputBufferDims();
    auto stepDims = getStepOutputBufferDims();
    BL_INFO("Type: {} -> {}", "N/A", TypeInfo<OT>::name);
    BL_INFO("Step Dimensions [A, F, T, P]: {} -> {}", "N/A", stepDims);
    if (config.numberOfTimeSampleStepsBeforeFrequencyChannelStep == 0) {
        BL_INFO("Stepping Time Samples First");
        gr_iterate.iterate_time_first_not_channel_first = true;
    }
    else if (config.numberOfTimeSampleStepsBeforeFrequencyChannelStep == 1) {
        BL_INFO("Stepping Frequency Channels First");
        gr_iterate.iterate_time_first_not_channel_first = false;
    }
    else {
        BL_INFO("Stepping Frequency Channels after every {} steps in Time Samples", config.numberOfTimeSampleStepsBeforeFrequencyChannelStep);
        gr_iterate.iterate_time_first_not_channel_first = true;
    }
    BL_INFO("Total Dimensions [A, F, T, P]: {} -> {} (across {} {})", "N/A", totalDims, this->gr_iterate.n_file, this->gr_iterate.n_file > 1 ? "files" : "file");
    BL_INFO("Steps in Dimensions [A, F, T, P]: {}", getNumberOfStepsInDimensions());
    if(config.requiredMultipleOfTimeSamplesSteps > 1) {
        BL_INFO("Rounded down to multiple of {} time steps", config.requiredMultipleOfTimeSamplesSteps);
    }
    BL_INFO("Input File Path: {}", config.filepath);
    
    // Allocate output buffers.
    BL_CHECK_THROW(output.stepDut1.resize({1}));
    BL_CHECK_THROW(output.stepJulianDate.resize({1}));
    BL_CHECK_THROW(output.stepFrequencyChannelOffset.resize({1}));
    BL_CHECK_THROW(output.stepBuffer.resize(stepDims));

    output.stepJulianDate[0] = getJulianDateOfLastReadBlock(this->config.stepNumberOfTimeSamples/2);
    BL_INFO("Starting Julian Date: {}", output.stepJulianDate[0]);
}

template<typename OT>
F64 Reader<OT>::getUnixDateOfLastReadBlock(const U64 timesamplesOffset) {
    return guppiraw_calc_unix_date(
        this->getChannelTimespan(),
        this->getDatashape()->n_time,
        getBlockMeta(&gr_iterate)->piperblk,
        getBlockMeta(&gr_iterate)->synctime,
        getBlockMeta(&gr_iterate)->pktidx // timesample index of first sample
        + this->lastread_block_index * getBlockMeta(&gr_iterate)->piperblk
        + timesamplesOffset
    );
}

template<typename OT>
const F64 Reader<OT>::getChannelBandwidth() const {
    return getBlockMeta(&gr_iterate)->chan_bw_mhz * 1e6;
}

template<typename OT>
const F64 Reader<OT>::getChannelTimespan() const {
    return getBlockMeta(&gr_iterate)->chan_timespan;
}

template<typename OT>
const F64 Reader<OT>::getObservationBandwidth() const {
    return getChannelBandwidth() * getBlockMeta(&gr_iterate)->fenchan;
}

template<typename OT>
const F64 Reader<OT>::getBandwidth() const {
    return getChannelBandwidth() * this->getDatashape()->n_aspectchan;
}

template<typename OT>
const U64 Reader<OT>::getChannelStartIndex() const {
    return getBlockMeta(&gr_iterate)->chan_start;
}

template<typename OT>
const F64 Reader<OT>::getCenterFrequency() const {
    return (double)getBlockMeta(&gr_iterate)->obs_freq_mhz * 1e6;
}

template<typename OT>
const F64 Reader<OT>::getObservationCenterFrequency() const {
    return getCenterFrequency() + (
        -((double)getChannelStartIndex())
        - (((double)this->getDatashape()->n_aspectchan) / 2.0)
        + (((double)getBlockMeta(&gr_iterate)->fenchan) / 2.0)
    ) * getChannelBandwidth();
}

template<typename OT>
const F64 Reader<OT>::getObservationBottomFrequency() const {
    return getCenterFrequency() + (
        -(double)getChannelStartIndex()
        - ((double)this->getDatashape()->n_aspectchan) / 2.0
    ) * getChannelBandwidth();
}

template<typename OT>
const F64 Reader<OT>::getBottomFrequency() const {
    return getCenterFrequency() + (
        - ((double)this->getDatashape()->n_aspectchan) / 2.0
    ) * getChannelBandwidth();
}

template<typename OT>
const F64 Reader<OT>::getObservationTopFrequency() const {
    return getCenterFrequency() + (
        -((double)getChannelStartIndex())
        - (((double)this->getDatashape()->n_aspectchan) / 2.0)
        + ((double)getBlockMeta(&gr_iterate)->fenchan)
    ) * getChannelBandwidth();
}

template<typename OT>
const F64 Reader<OT>::getTopFrequency() const {
    return getCenterFrequency() + (
        + (((double)this->getDatashape()->n_aspectchan) / 2.0)
    ) * getChannelBandwidth();
}

template<typename OT>
const F64 Reader<OT>::getAzimuthAngle() const {
    return getBlockMeta(&gr_iterate)->az;
}

template<typename OT>
const F64 Reader<OT>::getZenithAngle() const {
    return getBlockMeta(&gr_iterate)->el;
}

template<typename OT>
const F64 Reader<OT>::getRightAscension() const {
    return getBlockMeta(&gr_iterate)->ra_hour * BL_PHYSICAL_CONSTANT_PI / 12.0;
}

template<typename OT>
const F64 Reader<OT>::getDeclination() const {
    return getBlockMeta(&gr_iterate)->dec_deg *BL_PHYSICAL_CONSTANT_PI / 180.0;
}

template<typename OT>
const F64 Reader<OT>::getPhaseRightAscension() const {
    return getBlockMeta(&gr_iterate)->ra_phase_hour * BL_PHYSICAL_CONSTANT_PI / 12.0;
}

template<typename OT>
const F64 Reader<OT>::getPhaseDeclination() const {
    return getBlockMeta(&gr_iterate)->dec_phase_deg *BL_PHYSICAL_CONSTANT_PI / 180.0;
}

template<typename OT>
const std::string Reader<OT>::getSourceName() const {
    return std::string(getBlockMeta(&gr_iterate)->src_name);
}

template<typename OT>
const std::string Reader<OT>::getTelescopeName() const {
    return std::string(getBlockMeta(&gr_iterate)->telescope_id);
}

template<typename OT>
const Result Reader<OT>::preprocess(const cudaStream_t& stream,
                                    const U64& currentComputeCount) {
    if (!this->keepRunning()) {
        return Result::EXHAUSTED;
    }

    // Run library internal read method.
    const I64 bytes_read = 
        guppiraw_iterate_peek(&this->gr_iterate,
                              this->getStepOutputBufferDims().numberOfTimeSamples(),
                              this->getStepOutputBufferDims().numberOfFrequencyChannels(),
                              this->getStepOutputBufferDims().numberOfAspects(),
                              this->output.stepBuffer.data());

    if (bytes_read <= 0) {
        BL_FATAL("File reader couldn't read bytes.");
        return Result::ERROR;
    }

    this->lastread_block_index = gr_iterate.block_index;
    this->lastread_channel_index = gr_iterate.chan_index;
    this->lastread_time_index = gr_iterate.time_index;

    // Query internal library Unix Date, converting to Julian Date.
    this->output.stepJulianDate[0] = this->getJulianDateOfLastReadBlock(this->config.stepNumberOfTimeSamples/2);

    // Query internal library DUT1 value.
    this->output.stepDut1[0] = getBlockMeta(&gr_iterate)->dut1;

    // Stow frequency-channel offset
    this->output.stepFrequencyChannelOffset[0] = this->lastread_channel_index;


    if (config.numberOfTimeSampleStepsBeforeFrequencyChannelStep > 1) {
        // If stepping frequency-channels after N steps of time-samples,
        // step time first
        gr_iterate.iterate_time_first_not_channel_first = true;

        if (this->current_time_sample_step + 1 == config.numberOfTimeSampleStepsBeforeFrequencyChannelStep) {
            // unless this peeked step was the Nth time-sample step,
            // increment channel instead
            gr_iterate.iterate_time_first_not_channel_first = false; 
        }
    }

    bool fastestDimensionExhausted = guppiraw_iterate_increment(&this->gr_iterate,
                            this->getStepOutputBufferDims().numberOfTimeSamples(),
                            this->getStepOutputBufferDims().numberOfFrequencyChannels(),
                            this->getStepOutputBufferDims().numberOfAspects());
    
    if (config.numberOfTimeSampleStepsBeforeFrequencyChannelStep > 1) {
        if (gr_iterate.iterate_time_first_not_channel_first) {
            this->current_time_sample_step += 1;
            if (fastestDimensionExhausted) {
                BL_WARN("Time exhausted...");
            }
        }
        else {
            // just incremented channel instead
            this->current_time_sample_step = 0;

            if (fastestDimensionExhausted) {
                // wrapped on channel increment, so incremented time too
                // current time is checkpoint
                this->checkpoint_block_index = gr_iterate.block_index;
                this->checkpoint_time_index = gr_iterate.time_index;
            }
            else {
                // incremented channel, without wrapping
                // need to reset time to checkpoint for this
                // new channel
                guppiraw_iterate_set_time_index(&this->gr_iterate, this->checkpoint_block_index, this->checkpoint_time_index);
            }
        }   
    }
    else if (config.numberOfTimeSampleStepsBeforeFrequencyChannelStep == 0) {
        this->current_time_sample_step += 1;

        if (
            this->current_time_sample_step % this->config.requiredMultipleOfTimeSamplesSteps == 0
            && guppiraw_iterate_ntime_remaining(&this->gr_iterate) <
                this->config.requiredMultipleOfTimeSamplesSteps * this->getStepOutputBufferDims().numberOfTimeSamples()
        ) {
            // not enough time-steps remain,
            // reset time and increment channels once
            while(!fastestDimensionExhausted) {
                fastestDimensionExhausted = guppiraw_iterate_increment(&this->gr_iterate,
                                this->getStepOutputBufferDims().numberOfTimeSamples(),
                                this->getStepOutputBufferDims().numberOfFrequencyChannels(),
                                this->getStepOutputBufferDims().numberOfAspects());
            }
            this->current_time_sample_step = 0;
        }
    }

    return Result::SUCCESS;
}

template class BLADE_API Reader<CI8>;

}  // namespace Blade::Modules::Guppi
