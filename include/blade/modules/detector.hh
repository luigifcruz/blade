#ifndef BLADE_MODULES_DETECTOR_HH
#define BLADE_MODULES_DETECTOR_HH

#include <string>

#include "blade/base.hh"
#include "blade/module.hh"

namespace Blade {

enum class BLADE_API DetectorKernel : uint8_t {
    AFTP_1pol       = 0,
    AFTP_4pol       = 1,
    ATPF_1pol       = 2,
    ATPF_4pol       = 3,
    ATPFrev_1pol    = 4,
    ATPFrev_4pol    = 5,
};

constexpr const char* DetectorKernelName(const DetectorKernel kernel) {
    switch (kernel) {
        case DetectorKernel::AFTP_1pol:
            return "AFTP_1pol";
        case DetectorKernel::AFTP_4pol:
            return "AFTP_4pol";
        case DetectorKernel::ATPF_1pol:
            return "ATPF_1pol";
        case DetectorKernel::ATPF_4pol:
            return "ATPF_4pol";
        case DetectorKernel::ATPFrev_1pol:
            return "ATPFrev_1pol";
        case DetectorKernel::ATPFrev_4pol:
            return "ATPFrev_4pol";
        default:
            BL_FATAL("Detector kernel enumeration ({}) not supported.", 
                (int) kernel);
            BL_CHECK_THROW(Result::ERROR);
            return "?";
    }
}

} // namespace Blade

namespace Blade::Modules {

template<typename IT, typename OT>
class BLADE_API Detector : public Module {
 public:
    // Configuration

    struct Config {
        U64 integrationSize;
        DetectorKernel kernel = DetectorKernel::AFTP_4pol;

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

    // Constructor & Processing

    explicit Detector(const Config& config, const Input& input);
    const Result preprocess(const cudaStream_t& stream, const U64& currentComputeCount) final;
    const Result process(const cudaStream_t& stream) final;

 private:
    // Variables 

    const Config config;
    const Input input;
    Output output;

    U64 apparentIntegrationSize;
    Vector<Device::CUDA | Device::CPU, BOOL> ctrlResetTensor;

    // Expected Dimensions

    const ArrayDimensions getOutputBufferDims() const {
        return {
            .A = getInputBuffer().dims().numberOfAspects(),
            .F = getInputBuffer().dims().numberOfFrequencyChannels(),
            .T = getInputBuffer().dims().numberOfTimeSamples() / apparentIntegrationSize,
            .P = (U64) (((int)config.kernel) % 2 == 0 ? 1 : 4),
        }; 
    }
};

}  // namespace Blade::Modules

#endif

