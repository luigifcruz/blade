#ifndef BLADE_BENCHMARK_DETECTOR_GENERIC_HH
#define BLADE_BENCHMARK_DETECTOR_GENERIC_HH

#include "../../helper.hh"

namespace Blade {

template<template<typename, typename> class MUT, typename IT, typename OT>
class ModuleUnderTest : CudaBenchmark {
 public:
    const Result runComputeBenchmark(benchmark::State& state) {
        const U64 A = state.range(0);
        const U64 I = state.range(1);
        const U64 P = state.range(2);

        BL_CHECK(this->configureModule(I, P));
        BL_CHECK(this->allocateDeviceMemory(A));
        BL_CHECK(this->initializeModule());

        for (auto _ : state) {
            BL_CHECK(this->startIteration());
            
            {
                BL_CHECK(module->preprocess(this->getStream(), 0));
                BL_CHECK(module->process(this->getStream()));
            }

            BL_CHECK(this->finishIteration(state));
        }

        return Result::SUCCESS;
    }

    const Result runTransferBenchmark(benchmark::State& state) {
        const U64 A = state.range(0);

        BL_CHECK(this->allocateHostMemory(A));
        BL_CHECK(this->allocateDeviceMemory(A));

        for (auto _ : state) {
            BL_CHECK(this->startIteration());
            
            {
                BL_CHECK(Memory::Copy(deviceInputBuf, 
                                      hostInputBuf, 
                                      this->getStream()));
            }

            BL_CHECK(this->finishIteration(state));
        }

        return Result::SUCCESS;
    }

    const Result runConvergedBenchmark(benchmark::State& state) {
        const U64 A = state.range(0);
        const U64 I = state.range(1);
        const U64 P = state.range(2);

        BL_CHECK(this->configureModule(I, P));
        BL_CHECK(this->allocateHostMemory(A));
        BL_CHECK(this->allocateDeviceMemory(A));
        BL_CHECK(this->initializeModule());

        for (auto _ : state) {
            BL_CHECK(this->startIteration());
            
            {
                BL_CHECK(Memory::Copy(deviceInputBuf, 
                                      hostInputBuf, 
                                      this->getStream()));
                BL_CHECK(module->preprocess(this->getStream(), 0));
                BL_CHECK(module->process(this->getStream()));
            }

            BL_CHECK(this->finishIteration(state));
        }
        return Result::SUCCESS;
    }

protected:
    const Result configureModule(const U64& I, const U64& P) {
        config.integrationSize = I;
        config.numberOfOutputPolarizations = P;
        config.blockSize = 512;

        return Result::SUCCESS;
    }

    const Result allocateDeviceMemory(const U64& A) {
        BL_CHECK(deviceInputBuf.resize({
            .A = A,
            .F = 192,
            .T = 8192,
            .P = 2,
        }));

        return Result::SUCCESS;
    }

    const Result allocateHostMemory(const U64& A) {
        BL_CHECK(hostInputBuf.resize({
            .A = A,
            .F = 192,
            .T = 8192,
            .P = 2,
        }));

        return Result::SUCCESS;
    }

    const Result initializeModule() {
        BL_DISABLE_PRINT();
        Create(module, config, {
            .buf = deviceInputBuf, 
        }, this->getStream());
        BL_ENABLE_PRINT();

        return Result::SUCCESS;
    }

 private:
    MUT<IT, OT>::Config config;
    std::shared_ptr<MUT<IT, OT>> module;

    ArrayTensor<Device::CPU, IT> hostInputBuf;
    ArrayTensor<Device::CUDA, IT> deviceInputBuf;
};

}  // namespace Blade

#endif
