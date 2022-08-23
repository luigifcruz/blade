#ifndef BLADE_MODULES_CAST_GENERIC_HH
#define BLADE_MODULES_CAST_GENERIC_HH

#include "blade/base.hh"
#include "blade/module.hh"

namespace Blade::Modules {

template<typename IT, typename OT>
class BLADE_API Cast : public Module {
 public:
    struct Config {
        U64 inputSize;
        U64 blockSize = 512;
    };

    struct Input {
        const Vector<Device::CUDA, IT>& buf;
    };

    struct Output {
        Vector<Device::CUDA, OT> buf;
    };

    explicit Cast(const Config& config, const Input& input);

    constexpr Vector<Device::CUDA, IT>& getInput() {
        return const_cast<Vector<Device::CUDA, IT>&>(this->input.buf);
    }

    constexpr const Vector<Device::CUDA, OT>& getOutput() const {
        return this->output.buf;
    }

    constexpr const U64 getOutputSize() const {
        return this->config.inputSize;
    }

    constexpr const Config& getConfig() const {
        return this->config;
    }

    const Result process(const cudaStream_t& stream = 0) final;

 private:
    const Config config;
    const Input input;
    Output output;
};

}  // namespace Blade::Modules

#endif
