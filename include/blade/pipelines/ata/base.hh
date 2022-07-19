#ifndef BLADE_PIPELINES_BASE_ATA_HH
#define BLADE_PIPELINES_BASE_ATA_HH

#include "blade_config.hh"

#ifdef BLADE_PIPELINES_ATA_MODE_A
#include "blade/pipelines/ata/mode_a.hh"
#endif

#ifdef BLADE_PIPELINES_ATA_MODE_B
#include "blade/pipelines/ata/mode_B.hh"
#endif

#ifdef BLADE_PIPELINES_ATA_MODE_H
#include "blade/pipelines/ata/mode_h.hh"
#endif

#endif
