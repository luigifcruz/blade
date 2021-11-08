#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

typedef void* blade_module_t;

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
blade_module_t blade_initialize(size_t number_of_workers);

// Terminate the pipeline initialized by init().
//
// Parameters
// ----------
// mod : blade_module_t
//      pointer to the internal state
//
void blade_terminate(blade_module_t mod);

// Pin existing host memory in the device memory poll.
//
// Parameters
// ----------
// mod : blade_module_t
//      pointer to the internal state
// buffer : void*
//      buffer pointer
// size : size_t
//      buffer size in bytes
//
int blade_pin_memory(blade_module_t mod, void* buffer, size_t size);

// Get the expected size of the input buffer.
//
// Parameters
// ----------
// mod : blade_module_t
//      pointer to the internal state
//
size_t blade_get_input_size(blade_module_t mod);

// Get the expected size of the output buffer.
//
// Parameters
// ----------
// mod : blade_module_t
//      pointer to the internal state
//
size_t blade_get_output_size(blade_module_t mod);

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
int blade_process(blade_module_t mod, void** input, void** output);

// Submit a single buffer for asynchronous processing.
//
// This function is recommended over the synchronous counterpart.
//
// Parameters
// ----------
// mod : blade_module_t
//      pointer to the internal state
// idx : int
//      worker index
// input : void*
//      pointer of the input buffer (complex CI8)
// output : void*
//      pointer of the output buffer (complex CF16)
//
// Return
// ------
// int : error indicator (zero indicate success)
//
int blade_async_process(blade_module_t mod, int idx, void* input, void* output);

// Check if a worked finished processing a buffer.
//
// Parameters
// ----------
// mod : blade_module_t
//      pointer to the internal state
// idx : int
//      worker index
//
// Return
// ------
// bool : true if worker is done, otherwise false
//
bool blade_async_query(blade_module_t mod, int idx);
