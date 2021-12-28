#ifndef BLADE_RUNNER_HH
#define BLADE_RUNNER_HH

#include <deque>
#include <vector>
#include <memory>

#include "blade/logger.hh"
#include "blade/module.hh"
#include "blade/pipeline.hh"

namespace Blade {

class BLADE_API Runner {
 public:
    explicit Runner(const std::size_t& numberOfWorkers,
                    const std::function<const std::unique_ptr<Pipeline>>& workerInitializationFunc);
    virtual ~Runner();

    static Result SetCudaDevice(int device_id) {
        BL_CUDA_CHECK(cudaSetDevice(device_id), [&]{
           BL_FATAL("Failed to set device: {}", err);
        });
        return Result::SUCCESS;
    }

 private:
    const std::size_t numberOfWorkers;

    struct Job {
        std::unique_ptr<Pipeline>& worker;
        void* userData;
    };

    std::deque<Job> jobs;
    std::vector<std::unique_ptr<Pipeline>> workers;
};

}  // namespace Blade

#endif
