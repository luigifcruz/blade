#ifndef BLADE_BENCHMARK_CORRELATOR_GENERIC_HH
#define BLADE_BENCHMARK_CORRELATOR_GENERIC_HH

#include "../../helper.hh"
#include "blade/types.hh"

namespace Blade {

template<template<typename, typename> class MUT, typename IT, typename OT>
class CorrelatorTest : CudaBenchmark {
 public:
    typename MUT<IT, OT>::Config config;
    std::shared_ptr<MUT<IT, OT>> module;
    ArrayTensor<Device::CUDA, IT> deviceInputBuf;

    Result run(benchmark::State& state) {
        const U64 A = state.range(0);
        const U64 F = state.range(1);
        const U64 T = state.range(2);
        const U64 P = state.range(3);
        const U64 integrationRate = state.range(4);
        const U64 sharedMemory = state.range(5);
        const U64 calcMode = state.range(6);
        const U64 blockSize = state.range(7);

        state.SetLabel(bl::fmt::format("[{}, {}, {}, {}], "
                                       "Integration Size: {}, "
                                       "Shared Memory: {}, "
                                       "Calculation Mode: {}, "
                                       "Block Size: {}", A, F, T, P,
                                                         integrationRate,
                                                         sharedMemory,
                                                         calcMode,
                                                         blockSize));

        InitAndProfile([&](){
            config.integrationRate = integrationRate;
            config.useSharedMemory = sharedMemory;
            config.calculationMode = [&]{
                switch (calcMode) {
                    case 0: return CALC_MODE::INTEGER;
                    case 1: return CALC_MODE::SINGLE_PRECISION_FP;
                    case 2: return CALC_MODE::DOUBLE_PRECISION_FP;
                    default:
                        throw std::invalid_argument("Invalid calculation mode");
                }
            }();

            config.blockSize = blockSize;

            deviceInputBuf = ArrayTensor<Device::CUDA, IT>({A, F, T, P});

            BL_DISABLE_PRINT();
            Create(module, config, {
                .buf = deviceInputBuf,
            }, this->getStream());
            BL_ENABLE_PRINT();
        }, state);

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
