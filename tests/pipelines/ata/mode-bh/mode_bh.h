#ifndef MODE_BH_H
#define MODE_BH_H

#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#define BLADE_ATA_MODE_BH_NANT 20
#define BLADE_ATA_MODE_BH_NCHAN 192
#define BLADE_ATA_MODE_BH_NTIME 1024
#define BLADE_ATA_MODE_BH_NPOL 2
#define BLADE_ATA_MODE_BH_NBEAM 2
#define BLADE_ATA_MODE_BH_CHANNELIZER_RATE 1
#define BLADE_ATA_MODE_BH_ACCUMULATE_RATE 64
#define BLADE_ATA_MODE_BH_INTEGRATION_SIZE 2
#define BLADE_ATA_MODE_BH_INPUT_NCOMPLEX_BYTES 2
#define BLADE_ATA_MODE_BH_ITERATIONS 512
#define BLADE_ATA_MODE_BH_NUMBER_OF_WORKERS 2

typedef bool (blade_input_buffer_fetch_cb)(void*, void**);
typedef void (blade_input_buffer_ready_cb)(void*, const void*);
typedef bool (blade_output_buffer_fetch_cb)(void*, void**);
typedef void (blade_output_buffer_ready_cb)(void*, const void*);

bool blade_use_device(int device_id);
bool blade_pin_memory(void* buffer, size_t size);

void blade_ata_bh_register_user_data(void* user_data);
void blade_ata_bh_register_input_buffer_fetch_cb(blade_input_buffer_fetch_cb*  f);
void blade_ata_bh_register_input_buffer_ready_cb(blade_input_buffer_ready_cb* f);
void blade_ata_bh_register_output_buffer_fetch_cb(blade_output_buffer_fetch_cb* f);
void blade_ata_bh_register_output_buffer_ready_cb(blade_output_buffer_ready_cb* f);

bool blade_ata_bh_initialize(size_t numberOfWorkers);
size_t blade_ata_bh_get_input_size();
size_t blade_ata_bh_get_output_size();
void blade_ata_bh_compute_step();
void blade_ata_bh_terminate();

#endif
