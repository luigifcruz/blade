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
            /*
            PadGridSize(
                getInputBuffer().size(),
                config.blockSize
            ),
            */
            dim3( // grid dimensions (?)
                config.nAnts,
                2
            ),
            dim3( // threads per block
                160 //config.nChans
            ),
            0,
            // Kernel templates.
            TypeInfo<IT>::name,
            TypeInfo<OT>::name,
            config.debugMode,
            config.nAnts, 
            config.nChans, 
            8192, 
            2
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

    // Link output buffers.
    // if (config.inputPolarization == config.outputPolarization) {
    //     BL_INFO("Bypass: Enabled");
    // }

    // Link output buffer or link input with output.
    BL_CHECK_THROW(Link(output.buf, input.buf));
   
    // 32 comes from 8192 / 256
    // we are deciding to do blocks of 256 for kurtosis
    this->output.mask = ArrayTensor<Device::CUDA, U8>({config.nmaskruns, config.nAnts, config.nChans, 8192 / (4 * config.subblocksize)}, true);
    // this->output.mask = ArrayTensor<Device::CUDA, U8>({config.nmaskruns * config.nAnts, config.nChans, 8192, 2}, true);

    this->maskCounter = 0;
    this->maskOutFile.open("./blade_out.bin", std::ios::binary);
    
    // Print configuration values.
    BL_INFO("Type: {} -> {}", TypeInfo<IT>::name, TypeInfo<OT>::name);
    BL_INFO("Shape: {} -> {}", getInputBuffer().shape(), 
                              getOutputBuffer().shape());
    BL_INFO("Output Mask Shape: {}", getOutputMask().shape());
}

template<typename IT, typename OT>
Result Kurtosis<IT, OT>::writeMaskToDisk() {
    // mask write implementation

    int masksize = config.nmaskruns * config.nAnts * config.nChans * 8;
    // std::ofstream outputFile("/home/gsingh/temp/blade_out.bin", std::ios::out | std::ios::binary);
    // printf("writing from %p\n", this->output.mask);
    // for (int i = 0; i < masksize; i = i + 1) {
        /*
        U8 out = this->output.mask[i] + (this->output.mask[i + 1] << 1) + (this->output.mask[i + 2] << 2) + 
            (this->output.mask[i + 3] << 3) + (this->output.mask[i + 4] << 4) + (this->output.mask[i + 5] << 5) +
            (this->output.mask[i + 6] << 6) + (this->output.mask[i + 7] << 7);
        */
        // printf("%d\n", this->output.mask[i]);
        
        /*
        U8 out = this->output.mask[i] + this->output.mask[i + 1] + this->output.mask[i + 2] +
            this->output.mask[i + 3] + this->output.mask[i + 4] + this->output.mask[i + 5] + 
            this->output.mask[i + 6] + this->output.mask[i + 7];
        */

        // U8 out = this->output.mask[i] ;//+ this->output.mask[i + 2] + this->output.mask[i + 4] + this->output.mask[i + 6];

        // printf("%d %d\n", i, out);
        // this->maskOutFile.write(reinterpret_cast<const char*>(&out), sizeof(U8));;
    this->maskOutFile.write(reinterpret_cast<const char*>((this->output.mask.data())), masksize);

    // }
    // printf("done writing\n");
    // outputFile.close();
    return Result::SUCCESS;
}

template<typename IT, typename OT>
Kurtosis<IT, OT>::~Kurtosis() {
    // destructor
    if (this->maskCounter == config.nmaskruns) {
        this->writeMaskToDisk();
    }
    this->maskOutFile.close();
}

template<typename IT, typename OT>
Result Kurtosis<IT, OT>::process(const U64& currentStepCount, const Stream& stream) {
    /*
    if (config.inputPolarization == config.outputPolarization) {
        return Result::SUCCESS;
    }
    */

    // check mask flag/counter
    // if 0 nothing
    // else call this->writeMaskToDisk();

    if (this->maskCounter == config.nmaskruns) {
        /*
        while (this->future_obj.get() != Result::SUCCESS) {
            this->future_obj = std::async(this->writeMaskToDisk);
        }
        */
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
            /*
            getInputBuffer().shape().numberOfAspects(), 
            getInputBuffer().shape().numberOfFrequencyChannels(), 
            getInputBuffer().shape().numberOfTimeSamples(), 
            getInputBuffer().shape().numberOfPolarizations()
        );
        */
}

template class BLADE_API Kurtosis<CF32, CF32>;
template class BLADE_API Kurtosis<CF16, CF16>;

}  // namespace Blade::Modules
