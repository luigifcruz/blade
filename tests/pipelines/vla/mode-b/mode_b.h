#ifndef MODEB_H
#define MODEB_H

#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#define BLADE_VLA_MODE_B_CHANNELIZER_RATE 1048576 // [1, 4]; <= 1 mitigates the channelization

#define BLADE_VLA_MODE_B_INPUT_NANT 27
#define BLADE_VLA_MODE_B_INPUT_NCOMPLEX_BYTES 2

#define BLADE_VLA_MODE_B_ANT_NCHAN 1
#define BLADE_VLA_MODE_B_NTIME 1048576 
#define BLADE_VLA_MODE_B_NPOL 2

#define BLADE_VLA_MODE_B_OUTPUT_NBEAM 8
#define BLADE_VLA_MODE_B_OUTPUT_NCOMPLEX_BYTES 4

#define BLADE_VLA_MODE_B_OUTPUT_MEMCPY2D_PAD 0 // zero makes memcpy2D effectively 1D
#define BLADE_VLA_MODE_B_OUTPUT_MEMCPY2D_WIDTH 8192

#if BLADE_VLA_MODE_B_OUTPUT_NCOMPLEX_BYTES == 8
	#define BLADE_VLA_MODE_B_OUTPUT_ELEMENT_T CF32
#else
	#define BLADE_VLA_MODE_B_OUTPUT_ELEMENT_T CF16
#endif

bool blade_use_device(int device_id);
bool blade_vla_b_initialize(size_t numberOfWorkers);
size_t blade_vla_b_get_input_size();
size_t blade_vla_b_get_phasors_size();
size_t blade_vla_b_get_output_size();
bool blade_pin_memory(void* buffer, size_t size);
bool blade_vla_b_enqueue(void* input_ptr, void* phasors_ptr, void* output_ptr, size_t id);
bool blade_vla_b_dequeue(size_t* id);
void blade_vla_b_terminate();

#endif
