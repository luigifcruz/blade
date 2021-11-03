#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

typedef void* module_t;

// Initializes all modules from pipeline.
//
// Arguments
// ---------
// batch_size : size_t
//      specifies the number of parallel workers (usually higher than two)
//
// Return
// ------
// module_t : pointer to the internal state
//
module_t init(size_t batch_size);

// Destroys the module created by init().
//
// Arguments
// ---------
// mod : module_t
//      pointer to the internal state
//
void deinit(module_t mod);

// Pin host memory to the device poll.
//
// Arguments
// ---------
// mod : module_t
//      pointer to the internal state
// buffer : void*
//      buffer pointer
// size : size_t
//      size of the pointer
//
int pin_memory(module_t mod, void* buffer, size_t size);

// Get the expected size of each input buffer.
//
// Arguments
// ---------
// mod : module_t
//      pointer to the internal state
//
size_t get_input_size(module_t mod);

// Get the expected size of each output buffer.
//
// Arguments
// ---------
// mod : module_t
//      pointer to the internal state
//
size_t get_output_size(module_t mod);

// Process the data.
//
// Arguments
// ---------
// mod : module_t
//      pointer to the internal state
// input : void**
//      array of input buffers of size of batch_size (complex CI8)
// output : void**
//      array of output buffers of size of batch_size (complex CF16)
//
// Return
// ------
// int : error indicator (zero indicate success)
//
int process(module_t mod, void** input, void** output);

// Process a single buffer of data asynchronously.
//
// Arguments
// ---------
// mod : module_t
//      pointer to the internal state
// idx : int
//      index of the worker
// input : void*
//      array of input buffers of size of batch_size (complex CI8)
// output : void*
//      array of output buffers of size of batch_size (complex CF16)
//
// Return
// ------
// int : error indicator (zero indicate success)
//
int processAsync(module_t mod, int idx, void* input, void* output);

// Checks if a worker is done with the processing.
//
// Arguments
// ---------
// mod : module_t
//      pointer to the internal state
// idx : int
//      index of the worker
//
// Return
// ------
// bool : true if worker is done, otherwise false
//
bool synchronized(module_t mod, int idx);
