#include <benchmark/benchmark.h>

#include "blade/types.hh"

#ifdef BLADE_MODULE_CAST
#include "./cast/generic.hh"
#endif

#ifdef BLADE_MODULE_CHANNELIZER
#include "./channelizer/generic.hh"
#endif

#ifdef BLADE_MODULE_DETECTOR
#include "./detector/generic.hh"
#endif

#ifdef BLADE_MODULE_POLARIZER
#include "./polarizer/generic.hh"
#endif

#ifdef BLADE_MODULE_GATHER
#include "./gather/generic.hh"
#endif

#ifdef BLADE_MODULE_COPY
#include "./copy/generic.hh"
#endif

#ifdef BLADE_MODULE_PERMUTATION
#include "./permutation/generic.hh"
#endif

#ifdef BLADE_MODULE_ATA_BEAMFORMER
#include "./beamformer/ata.hh"
#endif

#ifdef BLADE_MODULE_MEERKAT_BEAMFORMER
#include "./beamformer/meerkat.hh"
#endif

#ifdef BLADE_MODULE_VLA_BEAMFORMER
#include "./beamformer/vla.hh"
#endif

BENCHMARK_MAIN();