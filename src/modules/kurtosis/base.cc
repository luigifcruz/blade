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
    // Configure kernel instantiation.
    BL_CHECK_THROW(
        this->createKernel(
            // Kernel name.
            std::string("main"),
            // Kernel function key.
            std::string("compute_sk_array"),
            // Kernel grid & block size.
            dim3( // grid dimensions
                getInputBuffer().shape().numberOfAspects(),
                2
            ),
            dim3( // threads per block
                160
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
            config.nKurtosisSigma,
            config.kurtosisBlockSize
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

    // Link output buffer or link input with output.
    BL_CHECK_THROW(Link(output.buf, input.buf));
   
    // 32 comes from 8192 / 256
    // we are deciding to do blocks of 256 for kurtosis
    this->output.mask = ArrayTensor<Device::CUDA, U8>({config.nMaskRuns, 
            getInputBuffer().shape().numberOfAspects(),
            getInputBuffer().shape().numberOfFrequencyChannels(),
            getInputBuffer().shape().numberOfTimeSamples() / (4 * config.kurtosisBlockSize)
            }, true);

    this->maskCounter = 0;
    this->maskOutFile.open(std::string(this->config.maskFilePath), std::ios::binary);

    // Print configuration values.
    BL_INFO("Type: {} -> {}", TypeInfo<IT>::name, TypeInfo<OT>::name);
    BL_INFO("Shape: {} -> {}", getInputBuffer().shape(), 
                              getOutputBuffer().shape());
    BL_INFO("Config: debugMode:          {}", this->config.debugMode);
    // BL_INFO("Config: nAnts:              {}", this->config.nAnts);
    // BL_INFO("Config: nPols:              {}", this->config.nPols);
    // BL_INFO("Config: nChans:             {}", this->config.nChans);
    BL_INFO("Config: kurtosisBlockSize:  {}", this->config.kurtosisBlockSize);
    BL_INFO("Config: nKurtosisSigma:     {}", this->config.nKurtosisSigma);
    BL_INFO("Config: nMaskRuns:          {}", this->config.nMaskRuns);
    BL_INFO("Config: maskFilePath:       {}", this->config.maskFilePath);
    BL_INFO("Output Mask Shape: {}", getOutputMask().shape());
}

template<typename IT, typename OT>
Result Kurtosis<IT, OT>::writeMaskToDisk() {
    // mask write implementation

    int masksize = config.nMaskRuns * getInputBuffer().shape().numberOfAspects() * getInputBuffer().shape().numberOfFrequencyChannels() * 8;
        
    this->maskOutFile.write(reinterpret_cast<const char*>((this->output.mask.data())), masksize);

    return Result::SUCCESS;
}

template<typename IT, typename OT>
Kurtosis<IT, OT>::~Kurtosis() {
    // destructor
    if (this->maskCounter == config.nMaskRuns) {
        this->writeMaskToDisk();
    }
    this->maskOutFile.close();
}

template<typename IT, typename OT>
Result Kurtosis<IT, OT>::process(const U64& currentStepCount, const Stream& stream) {

    BL_CHECK(Link(output.buf, input.buf));


    if (this->maskCounter == config.nMaskRuns) {
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
