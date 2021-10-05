#include "blade/kernels/beamformer.hh"

#include "blade/utils/magic_enum.hh"
#include "beamformer.jit.hh"

namespace Blade::Kernel {

Beamformer::Beamformer(const Config & config) : config(config), cache(100, *beamformer_kernel) {
    BL_DEBUG("Initilizating class.");

    if (config.NBEAMS > config.TBLOCK) {
        BL_FATAL("TBLOCK is smaller than NBEAMS.");
        throw Result::ERROR;
    }

    if ((config.NTIME % config.TBLOCK) != 0) {
        BL_FATAL("NTIME isn't divisable by TBLOCK.");
        throw Result::ERROR;
    }

    if (config.TBLOCK > 1024) {
        BL_FATAL("TBLOCK larger than hardware limit (1024).");
        throw Result::ERROR;
    }

    if ((config.TBLOCK % 32) != 0) {
        BL_WARN("Best performance is achieved when TBLOCK is a multiple of 32.");
    }

    block = dim3(config.TBLOCK);
    grid = dim3(config.NCHANS, config.NTIME/config.TBLOCK);

    kernel = Template(magic_enum::enum_name<Recipe>(config.recipe)).instantiate(
        config.NBEAMS,
        config.NANTS,
        config.NCHANS,
        config.NTIME,
        config.NPOLS,
        config.TBLOCK
    );
}

Beamformer::~Beamformer() {
    BL_DEBUG("Destroying class.");
}

Result Beamformer::run(const std::complex<int8_t>* input, const std::complex<float>* phasors,
        std::complex<float>* output) {

    cache.get_kernel(kernel)
        ->configure(grid, block)
        ->launch(
            reinterpret_cast<const char2*>(input),
            reinterpret_cast<const cuFloatComplex*>(phasors),
            reinterpret_cast<cuFloatComplex*>(output)
        );

    BL_CUDA_CHECK_KERNEL([&]{
        BL_FATAL("Kernel failed to execute: {}", err);
        return Result::CUDA_ERROR;
    });

    return Result::SUCCESS;
}

} // namespace Blade::Kernel
