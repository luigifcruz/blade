#ifndef BLADE_PIPELINES_BASE_HH
#define BLADE_PIPELINES_BASE_HH

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifndef BLADE_API
#define BLADE_API __attribute__((visibility("default")))
#endif

// Opaque pointer.
typedef void* blade_module_t;

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
bool BLADE_API blade_pin_memory(void* buffer, size_t size);

bool BLADE_API blade_use_device(int device_id);

#endif
