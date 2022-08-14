#ifndef MODE_BH_H
#define MODE_BH_H

#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#define BLADE_ATA_MODE_BH_CHANNELIZER_RATE 1

#define BLADE_ATA_MODE_BH_INPUT_NANT 28
#define BLADE_ATA_MODE_BH_INPUT_NCOMPLEX_BYTES 2

#define BLADE_ATA_MODE_BH_ANT_NCHAN 192
#define BLADE_ATA_MODE_BH_NTIME 8192
#define BLADE_ATA_MODE_BH_NPOL 2

#define BLADE_ATA_MODE_BH_OUTPUT_NBEAM 2

#define BLADE_ATA_MODE_BH_ACCUMULATE_RATE 64

bool blade_use_device(int device_id);
bool blade_ata_bh_initialize(size_t numberOfWorkers);
size_t blade_ata_bh_get_input_size();
size_t blade_ata_bh_get_output_size();
bool blade_pin_memory(void* buffer, size_t size);
bool blade_ata_bh_enqueue_b(void* input_ptr, const size_t id);
bool blade_ata_bh_dequeue_b(size_t* id);
bool blade_ata_bh_enqueue_h(const size_t b_id, void* output_ptr, const size_t h_id);
bool blade_ata_bh_dequeue_h(size_t* id);
void blade_ata_bh_terminate();

#endif
