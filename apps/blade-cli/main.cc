#include <CLI/CLI.hpp>
#include <iostream>
#include <string>

#include "blade/base.hh"
#include "blade/logger.hh"
#include "blade/runner.hh"
#include "blade/modules/guppi/reader.hh"
#include "blade/modules/bfr5/reader.hh"
#include "blade/pipelines/ata/mode_b.hh"

typedef enum {
    ATA,
    VLA,
    MEERKAT,
} TelescopeID;

typedef enum {
    MODE_B,
    MODE_A,
} ModeID;

using namespace Blade;

using CLIPipeline = Blade::Pipelines::ATA::ModeB<CF32>;
static std::unique_ptr<Runner<CLIPipeline>> runner;

template<typename OT = CI8>
class ReaderPipeline : public Pipeline {
 public:
    struct Config {
        std::string inputGuppiFile;
        std::string inputBfr5File; 

        U64 stepNumberOfTimeSamples;
        U64 stepNumberOfFrequencyChannels;
    };

    explicit ReaderPipeline(const Config& config) : config(config) {
        BL_DEBUG("Initializing CLI Reader Pipeline.");

        BL_DEBUG("Instantiating GUPPI RAW file reader.");
        this->connect(guppi, {
            .filepath = config.inputGuppiFile,
            .stepNumberOfTimeSamples = config.stepNumberOfTimeSamples, 
            .stepNumberOfFrequencyChannels = config.stepNumberOfFrequencyChannels,
        }, {});

        BL_DEBUG("Instantiating BFR5 file reader.");
        this->connect(bfr5, {
            .filepath = config.inputBfr5File,
        }, {});

        // Checking file and recipe bounds.

        if (guppi->getTotalNumberOfAntennas() != bfr5->getTotalNumberOfAntennas()) {
            BL_FATAL("Number of antennas from BFR5 ({}) and GUPPI RAW ({}) files mismatch.", 
                    guppi->getTotalNumberOfAntennas(), bfr5->getTotalNumberOfAntennas());
            BL_CHECK_THROW(Result::ASSERTION_ERROR);
        }

        if (guppi->getTotalNumberOfFrequencyChannels() != bfr5->getTotalNumberOfFrequencyChannels()) {
            BL_FATAL("Number of frequency channels from BFR5 ({}) and GUPPI RAW ({}) files mismatch.", 
                    guppi->getTotalNumberOfFrequencyChannels(), bfr5->getTotalNumberOfFrequencyChannels());
            BL_CHECK_THROW(Result::ASSERTION_ERROR);
        }

        if (guppi->getTotalNumberOfPolarizations() != bfr5->getTotalNumberOfPolarizations()) {
            BL_FATAL("Number of polarizations from BFR5 ({}) and GUPPI RAW ({}) files mismatch.", 
                    guppi->getTotalNumberOfPolarizations(), bfr5->getTotalNumberOfPolarizations());
            BL_CHECK_THROW(Result::ASSERTION_ERROR);
        }
   
        if (config.stepNumberOfFrequencyChannels != guppi->getTotalNumberOfFrequencyChannels()) {
            BL_WARN("Sub-band processing of the frequency channels ({}/{}) is not perfect yet.",
                config.stepNumberOfFrequencyChannels, guppi->getTotalNumberOfFrequencyChannels());
        }
    }

    const bool canRead() const {
        return guppi->canRead();
    }

    constexpr const U64 getNumberOfBeams() const {
        return bfr5->getTotalNumberOfBeams();
    }

    constexpr const U64 getNumberOfAntennas() const {
        return guppi->getNumberOfAntennas();
    }

    constexpr const U64 getNumberOfFrequencyChannels() const {
        return guppi->getNumberOfFrequencyChannels();
    }

    constexpr const U64 getNumberOfTimeSamples() const {
        return guppi->getNumberOfTimeSamples();
    }

    constexpr const U64 getNumberOfPolarizations() const {
        return guppi->getNumberOfPolarizations();
    }

    constexpr const F64 getObservationFrequency() const {
        return guppi->getObservationFrequency();
    }

    constexpr const F64 getChannelBandwidth() const {
        return guppi->getChannelBandwidth();
    }

    constexpr const F64 getTotalBandwidth() const {
        return guppi->getTotalBandwidth();
    }

    constexpr const U64 getChannelStartIndex() const {
        return guppi->getChannelStartIndex();
    }

    constexpr const LLA getReferencePosition() const {
        return bfr5->getReferencePosition();
    }

    constexpr const RA_DEC getBoresightCoordinate() const {
        return bfr5->getBoresightCoordinate();
    }

    constexpr const std::vector<XYZ> getAntennaPositions() const {
        return bfr5->getAntennaPositions();
    }

    constexpr const std::vector<CF64> getAntennaCalibrations(const U64& preBeamformerChannelizerRate) {
        return bfr5->getAntennaCalibrations(guppi->getNumberOfFrequencyChannels(), preBeamformerChannelizerRate);
    }

    constexpr const std::vector<RA_DEC> getBeamCoordinates() const {
        return bfr5->getBeamCoordinates();
    }

    constexpr Modules::Guppi::Reader<OT>& getGuppi() {
        return *guppi;
    }

 private:
    const Config config;

    std::shared_ptr<Modules::Guppi::Reader<OT>> guppi;
    std::shared_ptr<Modules::Bfr5::Reader> bfr5;
};

int main(int argc, char **argv) {

    CLI::App app("BLADE (Breakthrough Listen Accelerated DSP Engine) Command Line Tool");

    // Read target telescope. 

    TelescopeID telescope = TelescopeID::ATA;

    std::map<std::string, TelescopeID> telescopeMap = {
        {"ATA",     TelescopeID::ATA}, 
        {"VLA",     TelescopeID::VLA},
        {"MEERKAT", TelescopeID::MEERKAT}
    };

    app
        .add_option("-t,--telescope", telescope, "Telescope ID (ATA, VLA, MEETKAT)")
            ->required()
            ->transform(CLI::CheckedTransformer(telescopeMap, CLI::ignore_case));

    // Read target mode. 

    ModeID mode = ModeID::MODE_B;

    std::map<std::string, ModeID> modeMap = {
        {"MODE_B",     ModeID::MODE_B}, 
        {"MODE_A",     ModeID::MODE_A},
        {"B",          ModeID::MODE_B}, 
        {"A",          ModeID::MODE_A},
    };

    app
        .add_option("-m,--mode", mode, "Mode ID (MODE_B, MODE_A)")
            ->required()
            ->transform(CLI::CheckedTransformer(modeMap, CLI::ignore_case));

    // Read input GUPPI RAW file.

    std::string inputGuppiFile;

    app
        .add_option("-i,--input,input", inputGuppiFile, "Input GUPPI RAW filepath")
            ->required()
            ->capture_default_str()
            ->run_callback_for_default();

    // Read input BFR5 file.

    std::string inputBfr5File;

    app
        .add_option("-r,--recipe,recipe", inputBfr5File, "Input BFR5 filepath")
            ->required()
            ->capture_default_str()
            ->run_callback_for_default();

    // Read target step number of time samples.

    U64 numberOfTimeSamples = 32;

    app
        .add_option("-T,--time-samples", numberOfTimeSamples, "Step number of time samples")
            ->default_val(32);

    // Read target pre-beamformer channelizer rate.

    U64 preBeamformerChannelizerRate = 1024;

    app
        .add_option("-c,--pre-beamformer-channelizer-rate", preBeamformerChannelizerRate, 
            "Pre-beamformer channelizer rate (FFT-size)")
            ->default_val(1024);

    // Read target step number of frequency channels.

    U64 numberOfFrequencyChannels = 32;

    app
        .add_option("-C,--frequency-channels", numberOfFrequencyChannels, "Step number of frequency channels")
            ->default_val(32);

    // Parse arguments.

    CLI11_PARSE(app, argc, argv);

    // Print argument configurations.
    
    BL_INFO("Input GUPPI RAW File Path: {}", inputGuppiFile);
    BL_INFO("Input BFR5 File Path: {}", inputBfr5File);
    BL_INFO("Telescope: {}", telescope);
    BL_INFO("Mode: {}", mode);
    BL_INFO("Pre-beamformer channelizer rate: {}", preBeamformerChannelizerRate);

    auto reader = ReaderPipeline<CI8>({
        .inputGuppiFile = inputGuppiFile,
        .inputBfr5File = inputBfr5File,
        .stepNumberOfTimeSamples = numberOfTimeSamples * preBeamformerChannelizerRate,
        .stepNumberOfFrequencyChannels = numberOfFrequencyChannels,
    });

    const int numberOfWorkers = 1;
    switch (telescope) {
        case TelescopeID::ATA:
            switch (mode) {
                case ModeID::MODE_A:
                    BL_ERROR("Unsupported mode for ATA selected. WIP.");
                    break;
                case ModeID::MODE_B:
                    CLIPipeline::Config config = {
                        .preBeamformerChannelizerRate = preBeamformerChannelizerRate,

                        .phasorObservationFrequencyHz = reader.getObservationFrequency(),
                        .phasorChannelBandwidthHz = reader.getChannelBandwidth(),
                        .phasorTotalBandwidthHz = reader.getTotalBandwidth(),
                        .phasorFrequencyStartIndex = reader.getChannelStartIndex(),
                        .phasorReferenceAntennaIndex = 0,
                        .phasorArrayReferencePosition = reader.getReferencePosition(),
                        .phasorBoresightCoordinate = reader.getBoresightCoordinate(),
                        .phasorAntennaPositions = reader.getAntennaPositions(),
                        .phasorAntennaCalibrations = reader.getAntennaCalibrations(preBeamformerChannelizerRate),
                        .phasorBeamCoordinates = reader.getBeamCoordinates(),

                        .beamformerNumberOfAntennas = reader.getNumberOfAntennas(),
                        .beamformerNumberOfFrequencyChannels = reader.getNumberOfFrequencyChannels(),
                        .beamformerNumberOfTimeSamples = reader.getNumberOfTimeSamples(),
                        .beamformerNumberOfPolarizations = reader.getNumberOfPolarizations(),
                        .beamformerNumberOfBeams = reader.getNumberOfBeams(),
                        .beamformerIncoherentBeam = false,

                        .outputMemWidth = 8192,
                        .outputMemPad = 0,

                        .castBlockSize = 32,
                        .channelizerBlockSize = numberOfTimeSamples,
                        .phasorBlockSize = 32,
                        .beamformerBlockSize = numberOfTimeSamples
                    };
                    runner = Runner<CLIPipeline>::New(numberOfWorkers, config);
                    break;
            }
            break;
        default:
            BL_ERROR("Unsupported telescope selected. WIP");
            return 1;
    }

    Vector<Device::CPU, CF32>* output_buffers[numberOfWorkers];
    for (int i = 0; i < numberOfWorkers; i++) {
        output_buffers[i] = new Vector<Device::CPU, CF32>(runner->getWorker().getOutputSize());
        BL_INFO(
            "Allocated Runner output buffer {}: {} ({} bytes)",
            i,
            output_buffers[i]->size(),
            output_buffers[i]->size_bytes()
        );
    }

    U64 buffer_idx = 0;
    U64 job_idx = 0;

    while(reader.canRead()) {
        if (runner->enqueue(
        [&](auto& worker){
            worker.run(
                reader.getGuppi().getBlockEpochSeconds(reader.getGuppi().getNumberOfTimeSamples() / 2),
                0.0,
                reader.getGuppi().getOutput(),
                *output_buffers[buffer_idx]
            );
            return job_idx;
        }
        )) {
            buffer_idx = (buffer_idx + 1) % numberOfWorkers;
        }

        if (runner->dequeue(nullptr)) {
            job_idx++;
        }
    }

    runner.reset();

    return 0;
}
