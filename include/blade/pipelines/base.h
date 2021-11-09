#ifndef BLADE_PIPELINES_H
#define BLADE_PIPELINES_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifndef BLADE_API
#define BLADE_API __attribute__((visibility("default")))
#endif

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

#endif  // BLADE_INCLUDE_BLADE_META_H_
