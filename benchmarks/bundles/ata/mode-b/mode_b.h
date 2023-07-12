#ifndef MODE_B_H
#define MODE_B_H

#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#define DEBUG

#define BLADE_ATA_MODE_B_NANT 20
#define BLADE_ATA_MODE_B_NCHAN 192
#define BLADE_ATA_MODE_B_NTIME 8192
#define BLADE_ATA_MODE_B_NPOL 2
#define BLADE_ATA_MODE_B_NBEAM 2
#define BLADE_ATA_MODE_B_CHANNELIZER_RATE 1 
#define BLADE_ATA_MODE_B_ENABLE_INCOHERENT_BEAM true
#define BLADE_ATA_MODE_B_DETECTOR_ENABLED true
#define BLADE_ATA_MODE_B_DETECTOR_INTEGRATION 1
#define BLADE_ATA_MODE_B_DETECTOR_POLS 1
#define BLADE_ATA_MODE_B_INPUT_NCOMPLEX_BYTES 2
#define BLADE_ATA_MODE_B_OUTPUT_NCOMPLEX_BYTES 8
#define BLADE_ATA_MODE_B_NUMBER_OF_WORKERS 2

#define BLADE_ATA_MODE_B_INPUT_ELEMENT_T CI8
#if BLADE_ATA_MODE_B_OUTPUT_NCOMPLEX_BYTES == 8
    #if BLADE_ATA_MODE_B_DETECTOR_ENABLED 
        #define BLADE_ATA_MODE_B_OUTPUT_ELEMENT_T F32
    #else
        #define BLADE_ATA_MODE_B_OUTPUT_ELEMENT_T CF32
    #endif
#else
    #if BLADE_ATA_MODE_B_DETECTOR_ENABLED 
        #define BLADE_ATA_MODE_B_OUTPUT_ELEMENT_T F16
    #else
        #define BLADE_ATA_MODE_B_OUTPUT_ELEMENT_T CF16
    #endif
#endif

bool blade_ata_b_initialize(size_t numberOfWorkers);
size_t blade_ata_b_get_input_size();
size_t blade_ata_b_get_output_size();
bool blade_ata_b_enqueue(void* input_ptr, void* output_ptr, size_t id);
bool blade_ata_b_dequeue(size_t* id);
void blade_ata_b_terminate();

#endif
