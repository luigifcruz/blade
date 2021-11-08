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
    explicit Module(const typename T::Config& config) :
        Pipeline(false, true), config(config) {
        if (this->setup() != Result::SUCCESS) {
            throw Result::ERROR;
        }
    }

    Result run() {
        return this->loop();
    }

 protected:
    Result setupModules() final {
        BL_INFO("Initializing kernels.");

        beamformer = Factory<T>(config);

        return Result::SUCCESS;
    }

    Result setupMemory() final {
        BL_INFO("Allocating resources.");

        BL_CHECK(allocateBuffer(input, beamformer->getInputSize()));
        BL_CHECK(allocateBuffer(phasors, beamformer->getPhasorsSize()));
        BL_CHECK(allocateBuffer(output, beamformer->getOutputSize(), true));

        return Result::SUCCESS;
    }

    Result setupReport(Resources& res) final {
        BL_INFO("Reporting resources.");

        res.transfer.h2d += input.size_bytes();
        res.transfer.h2d += phasors.size_bytes();
        res.transfer.d2h += output.size_bytes();

        return Result::SUCCESS;
    }

    Result setupTest() final {
        test = std::make_unique<typename T::Test>(config);

        BL_CHECK(test->process());
        BL_CHECK(copyBuffer(input, test->getInputData(), CopyKind::H2D));
        BL_CHECK(copyBuffer(phasors, test->getPhasorsData(), CopyKind::H2D));

        return Result::SUCCESS;
    }

    Result loopProcess(cudaStream_t& cudaStream) final {
        BL_CHECK(beamformer->run(input, phasors, output, cudaStream));

        return Result::SUCCESS;
    }

    Result loopTest() final {
        Checker checker;

        std::size_t errors = 0;
        if ((errors = checker.run(output, test->getOutputData())) != 0) {
            BL_FATAL("Module produced {} errors.", errors);
            return Result::ERROR;
        }

        return Result::SUCCESS;
    }

 private:
    const typename T::Config& config;

    std::unique_ptr<Beamformer::Generic> beamformer;
    std::unique_ptr<Beamformer::Generic::Test> test;

    std::span<CF32> input;
    std::span<CF32> phasors;
    std::span<CF32> output;
};
