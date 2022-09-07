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
        const ArrayTensor<Device::CUDA, IT>& buf;
    };

    struct Output {
        ArrayTensor<Device::CUDA, OT> buf;
    };

    explicit Cast(const Config& config, const Input& input);

    constexpr const ArrayTensor<Device::CUDA, IT>& getInput() const {
        return this->input.buf;
    }

    constexpr const ArrayTensor<Device::CUDA, OT>& getOutput() const {
        return this->output.buf;
    }

    constexpr const Config& getConfig() const {
        return this->config;
    }

    const Result process(const cudaStream_t& stream = 0) final;

 private:
    const Config config;
    const Input input;
    Output output;

    constexpr const ArrayTensorDimensions getOutputDims() const {
        return getInput();
    }
};

}  // namespace Blade::Modules

#endif
