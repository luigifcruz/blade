#ifndef BLADE_PIPELINES_ATA_MODE_B_H
#define BLADE_PIPELINES_ATA_MODE_B_H

#include "blade/pipelines/base.h"

// Initialize the pipeline.
//
// Parameters
// ----------
// number_of_workers : size_t
//      specifies the number of workers to spawn (usually higher than two)
//
// Return
// ------
// blade_module_t : pointer to the internal state
//
blade_module_t BLADE_API blade_ata_b_initialize(size_t number_of_workers);

// Terminate the pipeline initialized by init().
//
// Parameters
// ----------
// mod : blade_module_t
//      pointer to the internal state
//
void BLADE_API blade_ata_b_terminate(blade_module_t mod);

// Get the expected size of a single input buffer.
//
// Parameters
// ----------
// mod : blade_module_t
//      pointer to the internal state
//
size_t BLADE_API blade_ata_b_get_input_size(blade_module_t mod);

// Get the expected size of a single output buffer.
//
// Parameters
// ----------
// mod : blade_module_t
//      pointer to the internal state
//
size_t BLADE_API blade_ata_b_get_output_size(blade_module_t mod);

// Submit a batch of buffers for synchronous processing.
//
// This function will block until all buffers are processed.
//
// Parameters
// ----------
// mod : blade_module_t
//      pointer to the internal state
// input : void**
//      array containing input buffers of size of number_of_workers (complex CI8)
// output : void**
//      array containing output buffers of size of number_of_workers (complex CF16)
//
// Return
// ------
// int : error indicator (zero indicate success)
//
int BLADE_API blade_ata_b_process(blade_module_t mod, void** input, void** output);


// Enqueue a single buffer for asynchronous processing, with the option to pass in 
// buffer IDs which can be retrieved in the `blade_ata_b_dequeue` call.
//
// Parameters
// ----------
// mod : blade_module_t
//      pointer to the internal state
// input : void*
//      input buffer (complex CI8)
// output : void*
//      output buffer (complex CF16)
// input_id : int
//      user ID for input buffer
// output_id : int
//      user ID for output buffer
//
// Return
// ------
// int : true if successfully enqueued, try again later if false
//
bool blade_ata_b_enqueue_with_id(blade_module_t mod, void* input, void* output,
                                 int input_id, int output_id);

// Enqueue a single buffer for asynchronous processing.
//
// Parameters
// ----------
// mod : blade_module_t
//      pointer to the internal state
// input : void*
//      input buffer (complex CI8)
// output : void*
//      output buffer (complex CF16)
//
// Return
// ------
// int : true if successfully enqueued, try again later if false
//
bool BLADE_API blade_ata_b_enqueue(blade_module_t mod, void* input, void* output);

// Dequeue a single buffer from asynchronous processing, with option to retrieve the 
// buffer IDs passed in during the `blade_ata_b_enqueue` call.
//
// Parameters
// ----------
// mod : blade_module_t
//      pointer to the internal state
// input : void**
//      optional, filled with the dequeued buffer input buffer (complex CI8)
// output : void**
//      optional, filled with the dequeued buffer output buffer (complex CF16)
// input_id : int*
//      optional, filled with the user supplied ID for input buffer
// output_id : int*
//      optional, filled with the user supplied ID for output buffer
//
// Return
// ------
// int : true if successfully dequeue, try again later if false
//
bool blade_ata_b_dequeue_with_id(blade_module_t mod, void** input, void** output,
                                 int* input_id, int* output_id);

// Dequeue a single buffer from asynchronous processing.
//
// Parameters
// ----------
// mod : blade_module_t
//      pointer to the internal state
// input : void**
//      optional, filled with the dequeued buffer input buffer (complex CI8)
// output : void**
//      optional, filled with the dequeued buffer output buffer (complex CF16)
//
// Return
// ------
// int : true if successfully dequeue, try again later if false
//
bool BLADE_API blade_ata_b_dequeue(blade_module_t mod, void** input, void** output);

#endif
