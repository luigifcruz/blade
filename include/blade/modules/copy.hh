#ifndef BLADE_MODULES_COPY_GENERIC_HH
#define BLADE_MODULES_COPY_GENERIC_HH

#include "blade/base.hh"
#include "blade/module.hh"

namespace Blade::Modules {

// MAYDO: Add built-in casting, if necessary.
// MAYDO: Add support for types different than ArrayTensor, if necessary.
template<typename IT, typename OT>
class BLADE_API Copy : public Module {
 public:
    // Configuration

    struct Config {
        U64 blockSize = 512;
    };

    constexpr const Config& getConfig() const {
        return this->config;
    }

    // Input

    struct Input {
        const ArrayTensor<Device::CUDA, IT>& buf;
    };

    constexpr const ArrayTensor<Device::CUDA, IT>& getInputBuffer() const {
        return this->input.buf;
    }

    // Output 

    struct Output {
        ArrayTensor<Device::CUDA, OT> buf;
    };

    constexpr const ArrayTensor<Device::CUDA, OT>& getOutputBuffer() const {
        return this->output.buf;
    }

    // Taint Registers

    constexpr Taint getTaint() const {
        return Taint::CONSUMER |
               Taint::PRODUCER;
    }

    // Constructor & Processing

    explicit Copy(const Config& config, const Input& input, 
                  const cudaStream_t& stream);
    Result process(const cudaStream_t& stream, const U64& currentStepCount) final;

 private:
    // Variables
        
    const Config config;
    const Input input;
    Output output;
};

}  // namespace Blade::Modules

#endif
