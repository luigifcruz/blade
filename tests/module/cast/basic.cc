#include "blade/modules/cast.hh"
#include "blade/utils/checker.hh"
#include "blade/memory.hh"
#include "blade/pipeline.hh"
#include "blade/memory.hh"

using namespace Blade;

template<typename IT, typename OT>
class Test : public Pipeline {
 public:
    explicit Test(const std::size_t& size) : input(size) {
        this->connect(cast, "cast", {512}, {input});
    }

    Result run(const Memory::HostVector<IT>& input,
                     Memory::HostVector<OT>& output) {
        BL_CHECK(this->copy(cast->getInput(), input));
        BL_CHECK(this->compute());
        BL_CHECK(this->copy(output, cast->getOutput()));
        BL_CHECK(this->synchronize());

        return Result::SUCCESS;
    }

 private:
    Memory::DeviceVector<IT> input;
    std::shared_ptr<Modules::Cast<IT, OT>> cast;
};

template<typename IT, typename OT>
int complex_test(const std::size_t testSize) {
    auto mod = Test<std::complex<IT>, std::complex<OT>>{testSize};

    Memory::HostVector<std::complex<IT>> input(testSize);
    Memory::HostVector<std::complex<OT>> output(testSize);
    Memory::HostVector<std::complex<OT>> result(testSize);

    for (std::size_t i = 0; i < testSize; i++) {
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

    std::size_t errors = 0;
    if ((errors = Checker::run(output, result)) != 0) {
        BL_FATAL("Cast produced {} errors.", errors);
        return 1;
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
