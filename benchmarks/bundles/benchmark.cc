#include <benchmark/benchmark.h>

#include "blade/types.hh"

#if defined(BLADE_BUNDLE_ATA_MODE_B)
#include "./ata/mode-b/generic.hh"
#endif

#if defined(BLADE_BUNDLE_ATA_MODE_B) && defined(BLADE_BUNDLE_ATA_MODE_H)
// TODO: Add Mode-BH benchmark.
//#include "./ata/mode-bh/generic.hh"
#endif

#if defined(BLADE_BUNDLE_GENERIC_MODE_H)
#include "./generic/mode-h/generic.hh"
#endif

BENCHMARK_MAIN();