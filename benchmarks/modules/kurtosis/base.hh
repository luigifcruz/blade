#ifndef BLADE_BENCHMARK_KURTOSIS_GENERIC_HH
#define BLADE_BENCHMARK_KURTOSIS_GENERIC_HH

#include "blade/modules/kurtosis.hh"

#include "../../helper.hh"

#include <random>

namespace Blade {

template<template<typename, typename> class MUT, typename IT, typename OT>
class KurtosisTest : CudaBenchmark {
 public:
    typename MUT<IT, OT>::Config config;
    std::shared_ptr<MUT<IT, OT>> module;
    ArrayTensor<Device::CUDA, IT> deviceInputBuf;

    // int nants = 42;
    // int nchans = 192;
    int nsamps = 8192;
    // int npols = 2; 

    Result run(benchmark::State& state) {
        // const U64 A = state.range(20);
        // const U8 M = state.range(1);

        InitAndProfile([&](){
            // config.inputPolarization = POL::XY;
            // config.outputPolarization = static_cast<POL>(M);
            // config.blockSize = 192;
            config.nAnts = 28;
            config.debugMode = true;

            deviceInputBuf = ArrayTensor<Device::CUDA, IT>({config.nAnts, config.nChans, nsamps, config.nPols}, true);

            // memset(&deviceInputBuf, 100, 28 * 192 * 8192 * 2);
            // printf("%.5f %.5f\n", deviceInputBuf[0].real(), deviceInputBuf[0].imag());

            //BL_DISABLE_PRINT();
            Create(module, config, {
                .buf = deviceInputBuf, 
            }, this->getStream());
            //BL_ENABLE_PRINT();
        }, state);

        /*
        for (int i = 0; i < config.nAnts * config.nChans; i++) {
            if ((rand() % 1000) < 1000) {
                int pol = rand() % 2;
                for (int s = 0; s < nsamps; s++) {
                    deviceInputBuf[(i * nsamps + s) * config.nPols + 0] = 100;
                    deviceInputBuf[(i * nsamps + s) * config.nPols + 1] = 100;
                }
            }
        }
        */

        for (int i = 0; i < config.nAnts; i++) {
            for (int j = 0; j < config.nChans; j++) {
                if ((rand() % 1000) < 5) {
                    int pol = rand() % 2;
                    for (int s = 0; s < nsamps; s++) {
                        deviceInputBuf[(((i * config.nChans) + j) * nsamps + s) * config.nPols + pol] = 100;
                        // deviceInputBuf[(((i * config.nChans) + j) * nsamps + s) * config.nPols + 1] = 100;
                    }
                }
            }
        }

        for (auto _ : state) {
            BL_CHECK(this->startIteration());
            BL_CHECK(module->process(0, this->getStream()));
            BL_CHECK(this->finishIteration(state));
        }

        return Result::SUCCESS;
    }
};

}  // namespace Blade

#endif
