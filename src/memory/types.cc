#include "blade/memory/types.hh"

namespace Blade {

template<>
const std::string TypeID<F16>() {
    return "F16";
}

template<>
const std::string TypeID<F32>() {
    return "F32";
}

template<>
const std::string TypeID<F64>() {
    return "F64";
}

template<>
const std::string TypeID<I8>() {
    return "I8";
}

template<>
const std::string TypeID<I16>() {
    return "I16";
}

template<>
const std::string TypeID<I32>() {
    return "I32";
}

template<>
const std::string TypeID<I64>() {
    return "I64";
}

template<>
const std::string TypeID<U8>() {
    return "U8";
}

template<>
const std::string TypeID<U16>() {
    return "U16";
}

template<>
const std::string TypeID<U32>() {
    return "U32";
}

template<>
const std::string TypeID<U64>() {
    return "U64";
}

template<>
const std::string TypeID<BOOL>() {
    return "BOOL";
}

template<>
const std::string TypeID<CF16>() {
    return "CF16";
}

template<>
const std::string TypeID<CF32>() {
    return "CF32";
}

template<>
const std::string TypeID<CF64>() {
    return "CF64";
}

template<>
const std::string TypeID<CI8>() {
    return "CI8";
}

template<>
const std::string TypeID<CI16>() {
    return "CI16";
}

template<>
const std::string TypeID<CI32>() {
    return "CI32";
}

template<>
const std::string TypeID<CI64>() {
    return "CI64";
}

template<>
const std::string TypeID<CU8>() {
    return "CU8";
}

template<>
const std::string TypeID<CU16>() {
    return "CU16";
}

template<>
const std::string TypeID<CU32>() {
    return "CU32";
}

template<>
const std::string TypeID<CU64>() {
    return "CU64";
}

}  // namespace Blade::Modules
