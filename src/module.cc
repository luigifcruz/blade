#include <typeindex>

#include "blade/module.hh"

namespace Blade {

Module::Module(const std::size_t& blockSize,
               const jitify2::PreprocessedProgram& kernel)
        : cache(100, *kernel),
          block(blockSize) {
    if (blockSize > 1024) {
        BL_FATAL("The block size ({}) is larger than hardware limit (1024).",
                blockSize);
        throw Result::ERROR;
    }

    if ((blockSize % 32) != 0) {
        BL_WARN("Best performance is achieved when the block size ({}) "
                "is a multiple of 32.", blockSize);
    }
}

template<typename T>
const std::string Module::cudaType() {
    static std::map<std::type_index, std::string> type_map = {
        {typeid(CF16),  "__half"},
        {typeid(CF32),  "float"},
        {typeid(CI8),   "char"},
        {typeid(CI16),  "short"},
        {typeid(CI32),  "long"},
        {typeid(F16),   "__half"},
        {typeid(F32),   "float"},
        {typeid(I8),    "char"},
        {typeid(I16),   "short"},
        {typeid(I32),   "long"},
    };
    return type_map[typeid(T)];
}

template const std::string Module::cudaType<CF16>();
template const std::string Module::cudaType<CF32>();
template const std::string Module::cudaType<CI8>();
template const std::string Module::cudaType<CI16>();
template const std::string Module::cudaType<CI32>();
template const std::string Module::cudaType<F16>();
template const std::string Module::cudaType<F32>();
template const std::string Module::cudaType<I8>();
template const std::string Module::cudaType<I16>();
template const std::string Module::cudaType<I32>();

template<typename T>
const std::size_t Module::cudaTypeSize() {
    static std::map<std::type_index, std::size_t> size_map = {
        {typeid(CF16),  2},
        {typeid(CF32),  2},
        {typeid(CI8),   2},
        {typeid(CI16),  2},
        {typeid(CI32),  2},
        {typeid(F16),   1},
        {typeid(F32),   1},
        {typeid(I8),    1},
        {typeid(I16),   1},
        {typeid(I32),   1},
    };
    return size_map[typeid(T)];
}

template const std::size_t Module::cudaTypeSize<CF16>();
template const std::size_t Module::cudaTypeSize<CF32>();
template const std::size_t Module::cudaTypeSize<CI8>();
template const std::size_t Module::cudaTypeSize<CI16>();
template const std::size_t Module::cudaTypeSize<CI32>();
template const std::size_t Module::cudaTypeSize<F16>();
template const std::size_t Module::cudaTypeSize<F32>();
template const std::size_t Module::cudaTypeSize<I8>();
template const std::size_t Module::cudaTypeSize<I16>();
template const std::size_t Module::cudaTypeSize<I32>();

}  // namespace Blade
