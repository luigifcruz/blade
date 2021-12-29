#ifndef BLADE_RUNNER_HH
#define BLADE_RUNNER_HH

#include <deque>
#include <vector>
#include <memory>

#include "blade/logger.hh"
#include "blade/module.hh"
#include "blade/pipeline.hh"

namespace Blade {

template<class T>
class BLADE_API Runner {
 public:
    static std::unique_ptr<Runner<T>> New(const std::size_t& numberOfWorkers,
                                          const typename T::Config& config) {
        return std::make_unique<Runner<T>>(numberOfWorkers, config);
    }

    explicit Runner(const std::size_t& numberOfWorkers,
                    const typename T::Config& config) {
        BL_INFO("Instantiating new runner.");

        if (numberOfWorkers == 0) {
            BL_FATAL("Number of worker has to be larger than zero.");
            BL_CHECK_THROW(Result::ASSERTION_ERROR);
        }

        for (std::size_t i = 0; i < numberOfWorkers; i++) {
            workers.push_back(std::make_unique<T>(config));
        }
    }

    virtual ~Runner() = default;

    constexpr const T& getWorker(const std::size_t& index = 0) const {
        return *workers[index];
    }

    constexpr const std::size_t& getHead() const {
        return head;
    }

    bool enqueue(const std::function<std::size_t(T&)>& jobFunc) {
        if (jobs.size() == workers.size()) {
            return false;
        }

        jobs.push_back({
            .id = jobFunc(*workers[head]),
            .worker = workers[head],
        });

        head = (head + 1) % workers.size();

        return true;
    }

    bool dequeue(std::size_t* id) {
        if (jobs.size() == 0) {
            return false;
        }

        const auto& job = jobs.front();

        if (jobs.size() == workers.size()) {
            job.worker->synchronize();
        }

        if (!job.worker->isSyncronized()) {
            return false;
        }

        *id = job.id;

        jobs.pop_front();

        return true;
    }

 private:
    struct Job {
        std::size_t id;
        std::unique_ptr<T>& worker;
    };

    std::size_t head = 0;
    std::deque<Job> jobs;
    std::vector<std::unique_ptr<T>> workers;
};

}  // namespace Blade

#endif
