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

    int nAnts = 42;
    int nChans = 192;
    int nSamps = 8192;
    int nPols = 2; 

    Result run(benchmark::State& state) {

        InitAndProfile([&](){
            config.debugMode = true;

            deviceInputBuf = ArrayTensor<Device::CUDA, IT>({42, 192, 8192, 2}, true);

            //BL_DISABLE_PRINT();
            Create(module, config, {
                .buf = deviceInputBuf, 
            }, this->getStream());
            //BL_ENABLE_PRINT();
        }, state);


        for (int i = 0; i < nAnts; i++) {
            for (int j = 0; j < nChans; j++) {
                if ((rand() % 1000) < 500) {
                    int pol = rand() % 2;
                    for (int s = 0; s < nSamps; s++) {
                        deviceInputBuf[(((i * nChans) + j) * nSamps + s) * nPols + pol] = 100;
                        deviceInputBuf[(((i * nChans) + j) * nSamps + s) * nPols + 1] = 100;
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
