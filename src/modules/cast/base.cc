#include <type_traits>
#include <typeindex>

#include "blade/modules/cast/base.hh"

#include "cast.jit.hh"

namespace Blade::Modules {

template<typename IT, typename OT>
Cast<IT, OT>::Cast(const Config& config, const Input& input)
        : Module(config.blockSize, cast_kernel),
          config(config),
          input(input) {

    // TODO: Add proper handling.
    std::map<std::type_index, std::string> mappy;

    mappy[typeid(CF32)] = "float";
    mappy[typeid(CF16)] = "__half";
    mappy[typeid(CI8)] = "char";

    kernel = Template("cast").instantiate(
        mappy[typeid(IT)],
        mappy[typeid(OT)],
        input.buf.size() * 2);

    grid = dim3(((input.buf.size() * 2) + block.x - 1) / block.x);
    block = dim3(config.blockSize);

    BL_CHECK_THROW(output.buf.allocate(input.buf.size()));
}

template<typename IT, typename OT>
Result Cast<IT, OT>::process(const cudaStream_t& stream) {
    cache
        .get_kernel(kernel)
        ->configure(grid, this->block, 0, stream)
        ->launch(input.buf, output.buf);

    BL_CUDA_CHECK_KERNEL([&]{
        BL_FATAL("Module failed to execute: {}", err);
        return Result::CUDA_ERROR;
    });

    return Result::SUCCESS;
}

template class Cast<CF32, CF16>;
template class Cast<CI8, CF32>;

}  // namespace Blade::Modules
