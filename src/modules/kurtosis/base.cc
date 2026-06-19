#define BL_LOG_DOMAIN "M::KURTOSIS"

#include <type_traits>
#include <typeindex>

#include <fstream>

#include "blade/modules/kurtosis.hh"

#include "kurtosis.jit.hh"

namespace Blade::Modules {

template<typename IT, typename OT>
Kurtosis<IT, OT>::Kurtosis(const Config& config, 
                             const Input& input, 
                             const Stream& stream)
        : Module(kurtosis_program),
          config(config),
          input(input) {
    
    const bool kurtosisStddevInRange = config.numberOfKurtosisStddev >= 3 || config.numberOfKurtosisStddev <= 9;
    if (!kurtosisStddevInRange) {
        BL_WARN("Kurtosis std-dev value expected to be with [3, 9], will default to 5 instead of {}.",
                 config.numberOfKurtosisStddev);
    }
    // Configure kernel instantiation.
    const unsigned int nthreads = 160; // inherited constant, not sure why, probably performance
    BL_CHECK_THROW(
        this->createKernel(
            // Kernel name.
            std::string("main"),
            // Kernel function key.
            std::string("compute_sk_array"),
            // Kernel grid & block size.
            dim3( // grid dimensions
                getInputBuffer().shape().numberOfAspects(),
                (getInputBuffer().shape().numberOfFrequencyChannels()+nthreads-1) / nthreads
            ),
            dim3( // threads per block
                nthreads
            ),
            0,
            // Kernel templates.
            TypeInfo<IT>::name,
            TypeInfo<OT>::name,
            config.debugMode,
            getInputBuffer().shape().numberOfAspects(),
            getInputBuffer().shape().numberOfFrequencyChannels(),
            getInputBuffer().shape().numberOfTimeSamples(), 
            getInputBuffer().shape().numberOfPolarizations(),
            config.numberOfKurtosisStddev,
            config.kurtosisChannelLength
        )
    );

    if constexpr (!std::is_same<IT, OT>::value) {
        BL_FATAL("This module requires the type of the input "
                 "({}) and output ({}) to be the same.",
                 TypeInfo<IT>::name, TypeInfo<OT>::name); 
        BL_INFO("Contact the maintainer if this "
                "functionality is required.");
        BL_CHECK_THROW(Result::ERROR);
    }
    
    if (getInputBuffer().shape().numberOfPolarizations() != 2) {
        BL_FATAL("Number of polarizations ({}) must be 2.",
                 getInputBuffer().shape().numberOfPolarizations());
        BL_CHECK_THROW(Result::ERROR);
    }
    
    const auto time_pols = (getInputBuffer().shape().numberOfTimeSamples() * getInputBuffer().shape().numberOfPolarizations());
    if (time_pols % (8 * config.kurtosisChannelLength) != 0) {
        BL_FATAL("Number of Timesample-Polarizations ({}) must be a whole multiple of 8 channel-lengths ({}).",
            time_pols,
            8 * config.kurtosisChannelLength
        );
        BL_CHECK_THROW(Result::ERROR);
    }

    // Link output buffer or link input with output.
    BL_CHECK_THROW(Link(output.buf, input.buf));
   
    // 32 comes from 8192 / 256
    // we are deciding to do blocks of 256 for kurtosis
    this->output.mask = ArrayTensor<Device::CUDA, U8>({config.numberOfMaskRuns, 
            getInputBuffer().shape().numberOfAspects(),
            getInputBuffer().shape().numberOfFrequencyChannels(),
            (getInputBuffer().shape().numberOfTimeSamples() * getInputBuffer().shape().numberOfPolarizations()) / (8 * config.kurtosisChannelLength)
            }, true);

    this->maskCounter = 0;
    this->maskOutFile.open(std::string(this->config.maskFilePath), std::ios::binary);

    // Print configuration values.
    BL_INFO("Type: {} -> {}", TypeInfo<IT>::name, TypeInfo<OT>::name);
    BL_INFO("Shape: {} -> {}", getInputBuffer().shape(), 
                              getOutputBuffer().shape());
    BL_INFO("Config: debugMode:          {}", this->config.debugMode);
    BL_INFO("Config: kurtosisChannelLength:  {}", this->config.kurtosisChannelLength);
    if (kurtosisStddevInRange) {
        BL_INFO("Config: numberOfKurtosisStddev: {}", this->config.numberOfKurtosisStddev);
    }
    else {
        BL_INFO("Config: numberOfKurtosisStddev: 5 ({} is out of range)", this->config.numberOfKurtosisStddev);
    }

    BL_INFO("Config: numberOfMaskRuns:       {}", this->config.numberOfMaskRuns);
    BL_INFO("Config: maskFilePath:           {}", this->config.maskFilePath);
    BL_INFO("Output Mask Shape: {}", getOutputMask().shape());
}

template<typename IT, typename OT>
Result Kurtosis<IT, OT>::writeMaskToDisk() {
    // mask write implementation
    this->maskOutFile.write(reinterpret_cast<const char*>((this->output.mask.data())), this->output.mask.size_bytes());

    return Result::SUCCESS;
}

template<typename IT, typename OT>
Kurtosis<IT, OT>::~Kurtosis() {
    // destructor
    if (this->maskCounter == config.numberOfMaskRuns) {
        this->writeMaskToDisk();
    }
    this->maskOutFile.close();
}

template<typename IT, typename OT>
Result Kurtosis<IT, OT>::process(const U64& currentStepCount, const Stream& stream) {

    BL_CHECK(Link(output.buf, input.buf));


    if (this->maskCounter == config.numberOfMaskRuns) {
        this->writeMaskToDisk();
        this->maskCounter = 0;
    }
    this->maskCounter += 1;
    
    return this->runKernel("main", 
            stream, 
            input.buf.data(),
            output.mask.data(),
            this->maskCounter
            );
}

template class BLADE_API Kurtosis<CF32, CF32>;
template class BLADE_API Kurtosis<CF16, CF16>;

}  // namespace Blade::Modules
