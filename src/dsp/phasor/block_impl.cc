#include <blade/phasor/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <blade/phasor/module.hh>

namespace Jetstream::Blocks {

struct PhasorImpl : public Block::Impl, public DynamicConfig<Blocks::Phasor> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::Phasor> moduleConfig = std::make_shared<Modules::Phasor>();
};

Result PhasorImpl::configure() {
    moduleConfig->observationFrequencyHz = observationFrequencyHz;
    moduleConfig->channelBandwidthHz = channelBandwidthHz;
    moduleConfig->totalBandwidthHz = totalBandwidthHz;
    moduleConfig->frequencyStartIndex = frequencyStartIndex;
    moduleConfig->referenceAntennaIndex = referenceAntennaIndex;
    moduleConfig->arrayReferenceLongitude = arrayReferenceLongitude;
    moduleConfig->arrayReferenceLatitude = arrayReferenceLatitude;
    moduleConfig->arrayReferenceAltitude = arrayReferenceAltitude;

    return Result::SUCCESS;
}

Result PhasorImpl::define() {
    JST_CHECK(defineInterfaceInput("antennaPositions", "Antenna Positions", "ECEF Antenna positions."));
    JST_CHECK(defineInterfaceInput("antennaCalibrations", "Antenna Calibrations", "Complex Antenna Calibrations (per channel and polarization)."));
    JST_CHECK(defineInterfaceInput("boresightCoordinates", "Boresight Coordinates", "Boresight RA and Dec."));
    JST_CHECK(defineInterfaceInput("beamCoordinates", "Beam Coordinates", "Beam coordinates (RA and Dec)."));
    JST_CHECK(defineInterfaceInput("julianDate", "Julian Date", "Julian date."));
    JST_CHECK(defineInterfaceInput("dut1", "DUT1", "Delta UT1."));
    JST_CHECK(defineInterfaceOutput("delays", "Delays", "Antenna delays for each beam."));
    JST_CHECK(defineInterfaceOutput("phasors", "Phasors", "Beamforming phasors."));

    JST_CHECK(defineInterfaceConfig("observationFrequencyHz",
                                    "Observation Frequency",
                                    "The observation center frequency.",
                                    "float:MHz:3"));
    JST_CHECK(defineInterfaceConfig("channelBandwidthHz",
                                    "Channel Bandwidth",
                                    "The channel bandwidth.",
                                    "float:MHz:3"));
    JST_CHECK(defineInterfaceConfig("totalBandwidthHz",
                                    "Total Bandwidth",
                                    "The total observation bandwidth. Such that channel 0 starts at `obs_center_freq - (total_bw / 2.0)`.",
                                    "float:MHz:3"));
    JST_CHECK(defineInterfaceConfig("frequencyStartIndex",
                                    "Frequency Start Index",
                                    "The zero-indexed frequency channel offset for the phasors.",
                                    "uint"));
    JST_CHECK(defineInterfaceConfig("referenceAntennaIndex",
                                    "Reference Antenna Index",
                                    "The index of the reference antenna.",
                                    "uint"));
    JST_CHECK(defineInterfaceConfig("arrayReferenceLongitude",
                                    "Array Reference Longitude",
                                    "The array reference position longitude.",
                                    "float:rad:3"));
    JST_CHECK(defineInterfaceConfig("arrayReferenceLatitude",
                                    "Array Reference Latitude",
                                    "The array reference position latitude.",
                                    "float:rad:3"));
    JST_CHECK(defineInterfaceConfig("arrayReferenceAltitude",
                                    "Array Reference Altitude",
                                    "The array reference position altitude.",
                                    "float:m:3"));

    return Result::SUCCESS;
}

Result PhasorImpl::create() {
    JST_CHECK(moduleCreate("phasor", moduleConfig, {
        {"antennaPositions", inputs().at("antennaPositions")},
        {"antennaCalibrations", inputs().at("antennaCalibrations")},
        {"boresightCoordinates", inputs().at("boresightCoordinates")},
        {"beamCoordinates", inputs().at("beamCoordinates")},
        {"julianDate", inputs().at("julianDate")},
        {"dut1", inputs().at("dut1")}
    }));
    JST_CHECK(moduleExposeOutput("delays", {"phasor", "delays"}));
    JST_CHECK(moduleExposeOutput("phasors", {"phasor", "phasors"}));

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(PhasorImpl, {"phasor"});

}  // namespace Jetstream::Blocks
