#ifndef BLADE_MEMORY_TYPES_HH
#define BLADE_MEMORY_TYPES_HH

#include <cuComplex.h>
#include <cuda_fp16.h>

#include <span>
#include <vector>
#include <string>

#include "blade/logger.hh"
#include "blade/types.hh"
#include "blade/macros.hh"

namespace Blade {

enum class BLADE_API Device : uint8_t {
    CPU     = 1 << 0,
    CUDA    = 1 << 1,
    METAL   = 1 << 2,
    VULKAN  = 1 << 3,
};

inline constexpr const Device operator|(Device lhs, Device rhs) {
    return static_cast<Device>(static_cast<uint8_t>(lhs) | static_cast<uint8_t>(rhs));
}

typedef __half   F16;
typedef float    F32;
typedef double   F64;
typedef int8_t   I8;
typedef int16_t  I16;
typedef int32_t  I32;
typedef int64_t  I64;
typedef uint8_t  U8;
typedef uint16_t U16;
typedef uint32_t U32;
typedef uint64_t U64;
typedef bool     BOOL;

typedef std::complex<F16> CF16;
typedef std::complex<F32> CF32;
typedef std::complex<F64> CF64;
typedef std::complex<I8>  CI8;
typedef std::complex<I16> CI16;
typedef std::complex<I32> CI32;
typedef std::complex<I64> CI64;
typedef std::complex<U8>  CU8;
typedef std::complex<U16> CU16;
typedef std::complex<U32> CU32;
typedef std::complex<U64> CU64;

template <typename T = void>
struct BLADE_API TypeInfo;

template<>
struct BLADE_API TypeInfo<F16> {
    using type = F16;
    using subtype = F16;
    inline static const std::string name = "F16";
};

template<>
struct BLADE_API TypeInfo<F32> {
    using type = F32;
    using subtype = F32;
    inline static const std::string name = "F32";
};

template<>
struct BLADE_API TypeInfo<F64> {
    using type = F64;
    using subtype = F64;
    inline static const std::string name = "F64";
};

template<>
struct BLADE_API TypeInfo<I8> {
    using type = I8;
    using subtype = I8;
    inline static const std::string name = "I8";
};

template<>
struct BLADE_API TypeInfo<I16> {
    using type = I16;
    using subtype = I16;
    inline static const std::string name = "I16";
};

template<>
struct BLADE_API TypeInfo<I32> {
    using type = I32;
    using subtype = I32;
    inline static const std::string name = "I32";
};

template<>
struct BLADE_API TypeInfo<I64> {
    using type = I64;
    using subtype = I64;
    inline static const std::string name = "I64";
};

template<>
struct BLADE_API TypeInfo<U8> {
    using type = U8;
    using subtype = U8;
    inline static const std::string name = "U8";
};

template<>
struct BLADE_API TypeInfo<U16> {
    using type = U16;
    using subtype = U16;
    inline static const std::string name = "U16";
};

template<>
struct BLADE_API TypeInfo<U32> {
    using type = U32;
    using subtype = U32;
    inline static const std::string name = "U32";
};

template<>
struct BLADE_API TypeInfo<U64> {
    using type = U64;
    using subtype = U64;
    inline static const std::string name = "U64";
};

template<>
struct BLADE_API TypeInfo<BOOL> {
    using type = BOOL;
    using subtype = BOOL;
    inline static const std::string name = "BOOL";
};

template<>
struct BLADE_API TypeInfo<CF16> {
    using type = CF16;
    using subtype = F16;
    inline static const std::string name = "CF16";
};

template<>
struct BLADE_API TypeInfo<CF32> {
    using type = CF32;
    using subtype = F32;
    inline static const std::string name = "CF32";
};

template<>
struct BLADE_API TypeInfo<CF64> {
    using type = CF64;
    using subtype = F64;
    inline static const std::string name = "CF64";
};

template<>
struct BLADE_API TypeInfo<CI8> {
    using type = CI8;
    using subtype = I8;
    inline static const std::string name = "CI8";
};

template<>
struct BLADE_API TypeInfo<CI16> {
    using type = CI16;
    using subtype = I16;
    inline static const std::string name = "CI16";
};

template<>
struct BLADE_API TypeInfo<CI32> {
    using type = CI32;
    using subtype = I32;
    inline static const std::string name = "CI32";
};

template<>
struct BLADE_API TypeInfo<CI64> {
    using type = CI64;
    using subtype = I64;
    inline static const std::string name = "CI64";
};

template<>
struct BLADE_API TypeInfo<CU8> {
    using type = CU8;
    using subtype = U8;
    inline static const std::string name = "CU8";
};

template<>
struct BLADE_API TypeInfo<CU16> {
    using type = CU16;
    using subtype = U16;
    inline static const std::string name = "CU16";
};

template<>
struct BLADE_API TypeInfo<CU32> {
    using type = CU32;
    using subtype = U32;
    inline static const std::string name = "CU32";
};

template<>
struct BLADE_API TypeInfo<CU64> {
    using type = CU64;
    using subtype = U64;
    inline static const std::string name = "CU64";
};

}  // namespace Blade

#endif
