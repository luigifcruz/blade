#ifndef MODEA_H
#define MODEA_H

#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#define BLADE_ATA_MODE_A_CHANNELIZER_RATE 4

#define BLADE_ATA_MODE_A_INPUT_NANT 20
#define BLADE_ATA_MODE_A_INPUT_NCOMPLEX_BYTES 2

#define BLADE_ATA_MODE_A_ANT_NCHAN 192
#define BLADE_ATA_MODE_A_NTIME 8192
#define BLADE_ATA_MODE_A_NPOL 2
#define BLADE_ATA_MODE_A_OUTPUT_NBEAM 1

#define BLADE_ATA_MODE_A_OUTPUT_MEMCPY2D_PAD 0
#define BLADE_ATA_MODE_A_OUTPUT_MEMCPY2D_WIDTH 8192

bool blade_use_device(int device_id);
bool blade_ata_a_initialize(size_t numberOfWorkers);
size_t blade_ata_a_get_input_size();
size_t blade_ata_a_get_output_size();
bool blade_pin_memory(void* buffer, size_t size);
bool blade_ata_a_enqueue(void* input_ptr, void* output_ptr, size_t id);
bool blade_ata_a_dequeue(size_t* id);
void blade_ata_a_terminate();

#endif
