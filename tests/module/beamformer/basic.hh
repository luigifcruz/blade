#include <memory>

#include "blade/beamformer/generic_test.hh"
#include "blade/beamformer/generic.hh"
#include "blade/checker/base.hh"
#include "blade/manager.hh"
#include "blade/pipeline.hh"

using namespace Blade;

template<typename T>
class Module : public Pipeline {
 public:
    explicit Module(const typename T::Config& config) : config(config) {
        if (this->commit() != Result::SUCCESS) {
            throw Result::ERROR;
        }
    }

 protected:
    Result underlyingInit() final {
        BL_INFO("Initializing kernels.");

        beamformer = Factory<T>(config);

        return Result::SUCCESS;
    }

    Result underlyingAllocate() final {
        BL_INFO("Allocating resources.");

        BL_CHECK(allocateBuffer(input, beamformer->getInputSize(), true));
        BL_CHECK(allocateBuffer(phasors, beamformer->getPhasorsSize(), true));
        BL_CHECK(allocateBuffer(output, beamformer->getOutputSize(), true));
        BL_CHECK(allocateBuffer(result, beamformer->getOutputSize(), true));

        BL_INFO("Generating test data.");
        for (auto& element : input) {
            element = 1;
        }

        for (auto& element : phasors) {
            element = 2.1;
        }

        for (auto& element : result) {
            element = config.dims.NANTS * 2.1;
        }

        return Result::SUCCESS;
    }

    Result underlyingReport(Resources& res) final {
        BL_INFO("Reporting resources.");

        res.transfer.h2d += input.size_bytes();
        res.transfer.h2d += phasors.size_bytes();
        res.transfer.d2h += output.size_bytes();

        return Result::SUCCESS;
    }

    Result underlyingProcess(cudaStream_t& cudaStream) final {
        BL_CHECK(beamformer->run(input, phasors, output, cudaStream));

        return Result::SUCCESS;
    }

    Result underlyingPostprocess() final {
        std::size_t errors = 0;
        if ((errors = checker.run(output, result)) != 0) {
            BL_FATAL("Module produced {} errors.", errors);
            return Result::ERROR;
        }

        return Result::SUCCESS;
    }

 private:
    const typename T::Config& config;

    std::unique_ptr<Beamformer::Generic> beamformer;

    Checker checker;

    std::span<CF32> input;
    std::span<CF32> phasors;
    std::span<CF32> output;
    std::span<CF32> result;
};
