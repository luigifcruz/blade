#ifndef BLADE_MODULES_BEAMFORMER_GENERIC_HH
#define BLADE_MODULES_BEAMFORMER_GENERIC_HH

#include <string>

#include "blade/base.hh"
#include "blade/module.hh"

namespace Blade::Modules::Beamformer {

template<typename IT, typename OT>
class BLADE_API Generic : public Module {
 public:
    // Configuration

    struct Config {
        BOOL enableIncoherentBeam = false;
        BOOL enableIncoherentBeamSqrt = false;

        U64 blockSize = 512;
    };

    constexpr const Config& getConfig() const {
        return config;
    }

    // Input

    struct Input {
        const ArrayTensor<Device::CUDA, IT>& buf;
        const PhasorTensor<Device::CUDA, IT>& phasors;
    };

    constexpr const ArrayTensor<Device::CUDA, IT>& getInputBuffer() const {
        return this->input.buf;
    }

    constexpr const PhasorTensor<Device::CUDA, IT>& getInputPhasors() const {
        return this->input.phasors;
    }

    // Output

    struct Output {
        ArrayTensor<Device::CUDA, OT> buf;
    };

    constexpr const ArrayTensor<Device::CUDA, OT>& getOutputBuffer() const {
        return this->output.buf;
    }

    // Taint Registers

    constexpr const MemoryTaint getMemoryTaint() {
        return MemoryTaint::CONSUMER | 
               MemoryTaint::PRODUCER;
    }

    // Constructor & Processing 

    explicit Generic(const Config& config, const Input& input,
                     const cudaStream_t& stream);
    virtual ~Generic() = default;
    const Result process(const cudaStream_t& stream) final;

 protected:
    // Variables

    const Config config;
    const Input input;
    Output output;

    // Expected Dimensions

    virtual const ArrayDimensions getOutputBufferDims() const = 0;
};

}  // namespace Blade::Modules::Beamformer

#endif
