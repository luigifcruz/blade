#include "blade/modules/cast.hh"
#include "blade/utils/checker.hh"
#include "blade/pipeline.hh"
#include "blade/memory/base.hh"

using namespace Blade;

template<typename IT, typename OT>
class Test : public Pipeline {
 public:
    explicit Test(const U64& inputSize) {
        BL_CHECK_THROW(input.resize({inputSize, 1, 1, 1}));
        this->connect(cast, {512}, {input});
    }

    const Result run(const ArrayTensor<Device::CPU, IT>& input,
                           ArrayTensor<Device::CPU, OT>& output) {
        BL_CHECK(this->copy(this->input, input));
        BL_CHECK(this->compute());
        BL_CHECK(this->copy(output, cast->getOutputBuffer()));
        BL_CHECK(this->synchronize());

        return Result::SUCCESS;
    }

 private:
    ArrayTensor<Device::CUDA, IT> input;
    std::shared_ptr<Modules::Cast<IT, OT>> cast;
};

template<typename IT, typename OT>
int complex_test(const U64 testSize) {
    auto mod = Test<std::complex<IT>, std::complex<OT>>{testSize};

    ArrayTensor<Device::CPU, std::complex<IT>> input({testSize, 1, 1, 1});
    ArrayTensor<Device::CPU, std::complex<OT>> output({testSize, 1, 1, 1});
    ArrayTensor<Device::CPU, std::complex<OT>> result({testSize, 1, 1, 1});

    for (U64 i = 0; i < testSize; i++) {
        input[i] = {
            static_cast<IT>(std::rand()),
            static_cast<IT>(std::rand())
        };

        result[i] = {
            static_cast<OT>(input[i].real()),
            static_cast<OT>(input[i].imag())
        };
    }

    for (int i = 0; i < 24; i++) {
        if (mod.run(input, output) != Result::SUCCESS) {
            BL_WARN("Fault was encountered. Test is exiting...");
            return 1;
        }
    }

    U64 errors = 0;
    if ((errors = Checker::run(output, result)) != 0) {
        BL_FATAL("Cast produced {} errors.", errors);
        return 1;
    }

    BL_INFO("Success...")

    return 0;
}

int main() {
    std::srand(unsigned(std::time(nullptr)));

    BL_INFO("Testing cast module.");

    const U64 testSize = 134400000;

    // TODO: Add non-complex tests.

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
