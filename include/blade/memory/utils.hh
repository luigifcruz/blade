#ifndef BLADE_MEMORY_UTILS_HH
#define BLADE_MEMORY_UTILS_HH

#include <cstdint>
#include <iostream>
#include <string>

namespace Blade::Memory {

static inline std::string ReadableBytes(uint64_t bytes) {
    const double GB = 1e9;
    const double MB = 1e6;
    const double KB = 1e3;

    char buffer[50];

    if (bytes >= GB) {
        sprintf(buffer, "%.2f GB", bytes / GB);
    } else if (bytes >= MB) {
        sprintf(buffer, "%.2f MB", bytes / MB);
    } else if (bytes >= KB) {
        sprintf(buffer, "%.2f KB", bytes / KB);
    } else {
        sprintf(buffer, "%ld bytes", bytes);
    }

    return std::string(buffer);
}

}  // namespace Blade::Memory

#endif
