#ifndef BLADE_MODULES_CAST_GENERIC_H
#define BLADE_MODULES_CAST_GENERIC_H

#include "blade/base.hh"
#include "blade/module.hh"

namespace Blade::Modules {

class BLADE_API Cast : public module {
 public:
    struct Config {
        std::size_t blockSize = 512;
    };

    explicit Cast(const Config& config);
    ~Cast();

    constexpr Config getConfig() const {
        return config;
    }

    template<typename IT, typename OT>
    Result run(const std::span<std::complex<IT>>& input,
                     std::span<std::complex<OT>>& output,
                     cudaStream_t cudaStream = 0);

 private:
    const Config config;
    dim3 block;
    jitify2::ProgramCache<> cache;

    template<typename OT, typename IT>
    Result run(IT input, OT output, std::size_t size,
            cudaStream_t cudaStream = 0);
};

}  // namespace Blade::Modules

#endif
