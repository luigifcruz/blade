#ifndef BLADE_PHASOR_MODULE_HH
#define BLADE_PHASOR_MODULE_HH

#include <jetstream/module.hh>

namespace Jetstream::Modules {

struct Phasor : public Module::Config {
    F64 observationFrequencyHz = 1.4e9;
    F64 channelBandwidthHz = 1e6;
    F64 totalBandwidthHz = 1e6;
    U64 frequencyStartIndex = 0;

    U64 referenceAntennaIndex = 0;
    F64 arrayReferenceLatitude = -40.00980599999271;
    F64 arrayReferenceLongitude = -52.874986919708704;
    F64 arrayReferenceAltitude = 818.6837417125288;

    JST_MODULE_TYPE(phasor);
    JST_MODULE_PARAMS(
        observationFrequencyHz,
        channelBandwidthHz,
        totalBandwidthHz,
        frequencyStartIndex,
        referenceAntennaIndex,
        arrayReferenceLatitude,
        arrayReferenceLongitude,
        arrayReferenceAltitude
    );
};

}  // namespace Jetstream::Modules

#endif  // BLADE_PHASOR_MODULE_HH
