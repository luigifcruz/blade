#ifndef BLADE_MEMORY_CPU_COPY_HH
#define BLADE_MEMORY_CPU_COPY_HH

#include "blade/memory/types.hh"
#include "blade/memory/vector.hh"

namespace Blade::Memory {

template<typename Type, typename Dims>
static const Result Copy(Vector<Device::CPU, Type, Dims>& dst,
                         const Vector<Device::CPU, Type, Dims>& src) {
    if (dst.size() != src.size()) {
        BL_FATAL("Size mismatch between source and destination ({}, {}).",
                src.size(), dst.size());
        return Result::ASSERTION_ERROR;
    }

    if (dst.dims() != src.dims()) {
        BL_FATAL("Dimensions mismatch between source ({}) and destination ({}).",
                src.dims(), dst.dims());
    }

    memcpy(dst.data(), src.data(), src.size_bytes());

    return Result::SUCCESS;
}

template<typename Type, typename Dims>
static const Result Copy(Vector<Device::CPU, Type, Dims>& dst,
                         const std::vector<Type>& src) {
    if (dst.size() != src.size()) {
        BL_FATAL("Size mismatch between source and destination ({}, {}).",
                src.size(), dst.size());
        return Result::ASSERTION_ERROR;
    }

    memcpy(dst.data(), src.data(), dst.size_bytes());

    return Result::SUCCESS;
}

template<typename Type, typename Dims>
static const Result Copy2D(Vector<Device::CPU, Type, Dims>& dst,
                           const U64& dstPitch,
                           const U64& dstOffset, 
                           const Vector<Device::CPU, Type, Dims>& src,
                           const U64& srcPitch,
                           const U64& srcOffset,
                           const U64& width_bytes,
                           const U64& height) {
    if (width_bytes > dstPitch) {
        BL_FATAL("2D copy 'width' is larger than destination's pitch ({}, {}).",
                width_bytes, dstPitch);
        return Result::ASSERTION_ERROR;
    }

    if (dst.size_bytes() < (dstPitch * height)) {
        BL_FATAL("Destination's size is exceeded by {} rows of {} ({} vs {}).",
                height, dstPitch, dst.size_bytes(), dstPitch * height);
        return Result::ASSERTION_ERROR;
    }

    if (width_bytes > srcPitch) {
        BL_FATAL("2D copy 'width' is larger than source's pitch ({}, {}).",
                width_bytes, srcPitch);
        return Result::ASSERTION_ERROR;
    }

    if (src.size_bytes() < (srcPitch * height)) {
        BL_FATAL("Source's size is exceeded by {} rows of {} ({} vs {}).",
                height, srcPitch, src.size_bytes(), srcPitch * height);
        return Result::ASSERTION_ERROR;
    }

    if (width_bytes % sizeof(Type) != 0) {
        BL_FATAL("2D copy 'width' is not a multiple of element size ({}, {}).",
                width_bytes, sizeof(Type));
        return Result::ASSERTION_ERROR;
    }

    if (srcOffset % sizeof(Type) != 0) {
        BL_FATAL("2D copy source offset (bytes) is not a multiple of element size ({}, {}).",
                srcOffset, sizeof(Type));
        return Result::ASSERTION_ERROR;
    }

    auto dst_data = dst.data() + dstOffset;
    auto src_data = src.data() + srcOffset;

    for (U64 i = 0; i < height; i++) {
        memcpy(
            dst_data,
            src_data,
            width_bytes
        );
        dst_data += dstPitch;
        src_data += srcPitch;
    }

    return Result::SUCCESS;
}

}  // namespace Blade::Memory

#endif
