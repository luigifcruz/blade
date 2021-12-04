#include "blade/modules/beamformer/ata.hh"
#include "blade/utils/checker.hh"

using namespace Blade;

template<typename IT, typename OT>
class Test : public Pipeline {
 public:
    explicit Test(const typename Modules::Beamformer::ATA<IT, OT>::Config& config) {
        this->connect(beamformer, config, {input, phasors});
    }

    constexpr const std::size_t getInputSize() const {
        return beamformer->getInputSize();
    }

    constexpr const std::size_t getPhasorsSize() const {
        return beamformer->getPhasorsSize();
    }

    constexpr const std::size_t getOutputSize() const {
        return beamformer->getOutputSize();
    }

    Result run(const Memory::HostVector<IT>& input,
               const Memory::HostVector<IT>& phasors,
                     Memory::HostVector<OT>& output) {
        BL_CHECK(this->copy(beamformer->getInput(), input));
        BL_CHECK(this->copy(beamformer->getPhasors(), phasors));
        BL_CHECK(this->compute());
        BL_CHECK(this->copy(output, beamformer->getOutput()));
        BL_CHECK(this->synchronize());

        return Result::SUCCESS;
    }

 private:
    Memory::DeviceVector<IT> input;
    Memory::DeviceVector<IT> phasors;
    std::shared_ptr<Modules::Beamformer::ATA<IT, OT>> beamformer;
};
