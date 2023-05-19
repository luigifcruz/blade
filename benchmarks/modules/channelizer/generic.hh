#ifndef BLADE_BENCHMARK_CHANNELIZER_GENERIC_HH
#define BLADE_BENCHMARK_CHANNELIZER_GENERIC_HH

#include "../../helper.hh"

namespace Blade {

template<template<typename, typename> class MUT, typename IT, typename OT>
class ModuleUnderTest : CudaBenchmark {
 public:
    Result runComputeBenchmark(benchmark::State& state) {
        const U64 A = state.range(0);
        const U64 R = state.range(1);

        BL_CHECK(InitAndProfile([&](){
            BL_CHECK(this->configureModule(R));
            BL_CHECK(this->allocateDeviceMemory(A));
            BL_CHECK(this->initializeModule());
            return Result::SUCCESS;
        }, state));

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

    Result runTransferBenchmark(benchmark::State& state) {
        const U64 A = state.range(0);

        BL_CHECK(InitAndProfile([&](){
            BL_CHECK(this->allocateHostMemory(A));
            BL_CHECK(this->allocateDeviceMemory(A));
            return Result::SUCCESS;
        }, state));

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

    Result runConvergedBenchmark(benchmark::State& state) {
        const U64 A = state.range(0);
        const U64 R = state.range(1);

        BL_CHECK(InitAndProfile([&](){
            BL_CHECK(this->configureModule(R));
            BL_CHECK(this->allocateHostMemory(A));
            BL_CHECK(this->allocateDeviceMemory(A));
            BL_CHECK(this->initializeModule());
            return Result::SUCCESS;
        }, state));
        
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
    Result configureModule(const U64& R) {
        config.rate = R;
        config.blockSize = 512;

        return Result::SUCCESS;
    }

    Result allocateDeviceMemory(const U64& A) {
        deviceInputBuf = ArrayTensor<Device::CUDA, IT>({A, 192, 8192, 2});

        return Result::SUCCESS;
    }

    Result allocateHostMemory(const U64& A) {
        hostInputBuf = ArrayTensor<Device::CPU, IT>({A, 192, 8192, 2});

        return Result::SUCCESS;
    }

    Result initializeModule() {
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
