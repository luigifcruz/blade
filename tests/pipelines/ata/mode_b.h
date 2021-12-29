#ifndef MODEB_H
#define MODEB_H

#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>

bool blade_use_device(int device_id);
bool blade_ata_b_initialize(size_t numberOfWorkers);
size_t blade_ata_b_get_input_size();
size_t blade_ata_b_get_output_size();
bool blade_pin_memory(void* buffer, size_t size);
bool blade_ata_b_enqueue(void* input_ptr, void* output_ptr, size_t id);
bool blade_ata_b_dequeue(size_t* id);
void blade_ata_b_terminate();

#endif
