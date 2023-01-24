#ifndef BLADE_MODULES_BASE_HH
#define BLADE_MODULES_BASE_HH

#include "blade_config.hh"

#ifdef BLADE_MODULE_CAST
#include "blade/modules/cast.hh"
#endif

#ifdef BLADE_MODULE_CHANNELIZER
#include "blade/modules/channelizer.hh"
#endif

#ifdef BLADE_MODULE_DETECTOR
#include "blade/modules/detector.hh"
#endif

#ifdef BLADE_MODULE_POLARIZER
#include "blade/modules/polarizer.hh"
#endif

#ifdef BLADE_MODULE_GUPPI
#include "blade/modules/guppi/base.hh"
#endif

#ifdef BLADE_MODULE_BFR5
#include "blade/modules/bfr5/base.hh"
#endif

#ifdef BLADE_MODULE_FILTERBANK
#include "blade/modules/filterbank/base.hh"
#endif

#ifdef BLADE_MODULE_ATA_BEAMFORMER
#include "blade/modules/beamformer/ata.hh"
#endif

#ifdef BLADE_MODULE_ATA_PHASOR
#include "blade/modules/phasor/ata.hh"
#endif

#endif
