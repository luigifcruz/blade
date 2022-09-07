#ifndef BLADE_MODULES_BEAMFORMER_GENERIC_HH
#define BLADE_MODULES_BEAMFORMER_GENERIC_HH

#include <string>

#include "blade/base.hh"
#include "blade/module.hh"

namespace Blade::Modules::Beamformer {

template<typename IT, typename OT>
class BLADE_API Generic : public Module {
 public:
    struct Config {
        BOOL enableIncoherentBeam = false;
        BOOL enableIncoherentBeamSqrt = false;

        U64 blockSize = 512;
    };

    struct Input {
        const ArrayTensor<Device::CUDA, IT>& buf;
        const PhasorTensor<Device::CUDA, IT>& phasors;
    };

    struct Output {
        ArrayTensor<Device::CUDA, OT> buf;
    };

    explicit Generic(const Config& config, const Input& input);
    virtual ~Generic() = default;

    constexpr const ArrayTensor<Device::CUDA, IT>& getInputBuffer() const {
        return this->input.buf;
    }

    constexpr const PhasorTensor<Device::CUDA, IT>& getInputPhasors() const {
        return this->input.phasors;
    }

    constexpr const ArrayTensor<Device::CUDA, OT>& getOutput() const {
        return this->output.buf;
    }

    constexpr Config getConfig() const {
        return config;
    }

    const Result process(const cudaStream_t& stream = 0) final;

 protected:
    const Config config;
    const Input input;
    Output output;

    virtual constexpr const ArrayTensorDimensions getOutputDims() const = 0;
};

}  // namespace Blade::Modules::Beamformer

#endif
