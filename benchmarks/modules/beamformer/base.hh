#ifndef BLADE_BENCHMARK_BEAMFORMER_GENERIC_HH
#define BLADE_BENCHMARK_BEAMFORMER_GENERIC_HH

#include "../../helper.hh"

namespace Blade {

template<template<typename, typename> class MUT, typename IT, typename OT>
class BeamformerTest : CudaBenchmark {
 public:
    typename MUT<IT, OT>::Config config;
    std::shared_ptr<MUT<IT, OT>> module;

    ArrayTensor<Device::CUDA, IT> deviceInputBuf;
    PhasorTensor<Device::CUDA, IT> deviceInputPhasors;

    Result run(benchmark::State& state) {
        const U64 A = state.range(1);
        const U64 B = state.range(0);

        InitAndProfile([&](){
            config.enableIncoherentBeam = false;
            config.enableIncoherentBeamSqrt = false;
            config.blockSize = 512;

            deviceInputBuf = ArrayTensor<Device::CUDA, IT>({A, 192, 8192, 2});
            deviceInputPhasors = PhasorTensor<Device::CUDA, IT>({B, A, 192, 1, 2});

            BL_DISABLE_PRINT();
            Create(module, config, {
                .buf = deviceInputBuf, 
                .phasors = deviceInputPhasors,
            }, this->getStream());
            BL_ENABLE_PRINT();
        }, state);

        for (auto _ : state) {
            BL_CHECK(this->startIteration());
            BL_CHECK(module->process(this->getStream(), 0));
            BL_CHECK(this->finishIteration(state));
        }

        return Result::SUCCESS;
    }
};

}  // namespace Blade

#endif
