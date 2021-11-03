#include "blade/cast/base.hh"
#include "blade/checker/base.hh"
#include "blade/manager.hh"
#include "blade/pipeline.hh"

using namespace Blade;

template<typename IT, typename OT>
class Module : public Pipeline {
 public:
    explicit Module(const std::size_t& size) : size(size) {
        if (this->setup() != Result::SUCCESS) {
            throw Result::ERROR;
        }
    }

    Result run() {
        return this->loop(false);
    }

    Result loadTestData(const std::span<IT>& inputBuffer,
                  const std::span<OT>& resultBuffer) {
        BL_CHECK(copyBuffer(input, inputBuffer, CopyKind::H2D));
        BL_CHECK(copyBuffer(result, resultBuffer, CopyKind::H2D));

        return Result::SUCCESS;
    }

 protected:
    Result setupModules() final {
        BL_INFO("Initializing kernels.");

        cast = Factory<Cast>({});

        return Result::SUCCESS;
    }

    Result setupMemory() final {
        BL_INFO("Allocating resources.");

        BL_CHECK(allocateBuffer(input, size));
        BL_CHECK(allocateBuffer(output, size, true));
        BL_CHECK(allocateBuffer(result, size, true));

        return Result::SUCCESS;
    }

    Result setupReport(Resources& res) final {
        BL_INFO("Reporting resources.");

        res.transfer.h2d += input.size_bytes();
        res.transfer.h2d += result.size_bytes();

        return Result::SUCCESS;
    }

    Result loopProcess(cudaStream_t& cudaStream) final {
        BL_CHECK(cast->run(input, output, cudaStream));

        return Result::SUCCESS;
    }

    Result loopPostprocess() final {
        std::size_t errors = 0;
        if ((errors = checker.run(output, result)) != 0) {
            BL_FATAL("Module produced {} errors.", errors);
            return Result::ERROR;
        }

        return Result::SUCCESS;
    }

 private:
    const std::size_t size;

    std::span<IT> input;
    std::span<OT> output;
    std::span<OT> result;

    std::unique_ptr<Cast> cast;
    Checker checker;
};

template<typename IT, typename OT>
int complex_test(const std::size_t testSize) {
    Manager manager{};
    Module<std::complex<IT>, std::complex<OT>> mod{testSize};

    std::vector<std::complex<IT>> input;
    std::vector<std::complex<OT>> result;
    for (std::size_t i = 0; i < testSize; i++) {
        input.push_back({
            static_cast<IT>(std::rand()),
            static_cast<IT>(std::rand())
        });

        result.push_back({
            static_cast<OT>(input[i].real()),
            static_cast<OT>(input[i].imag())
        });
    }
    mod.loadTestData(input, result);
    manager.save(mod).report();

    for (int i = 0; i < 24; i++) {
        if (mod.run() != Result::SUCCESS) {
            BL_WARN("Fault was encountered. Test is exiting...");
            return 1;
        }
    }

    BL_INFO("Success...")

    return 0;
}

int main() {
    Logger guard{};
    std::srand(unsigned(std::time(nullptr)));

    BL_INFO("Testing cast module.");

    const std::size_t testSize = 134400000;

    BL_INFO("Casting CI8 to CF32...");
    if (complex_test<I8, F32>(testSize) != 0) {
        return 1;
    }

    BL_INFO("Casting CF32 to CF16...");
    if (complex_test<F32, F16>(testSize) != 0) {
        return 1;
    }

    BL_INFO("Test succeeded.");

    return 0;
}
