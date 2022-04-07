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
    static std::unique_ptr<Runner<T>> New(const U64& numberOfWorkers,
                                          const typename T::Config& config) {
        return std::make_unique<Runner<T>>(numberOfWorkers, config);
    }

    explicit Runner(const U64& numberOfWorkers,
                    const typename T::Config& config) {
        BL_INFO("Instantiating new runner.");

        if (numberOfWorkers == 0) {
            BL_FATAL("Number of worker has to be larger than zero.");
            BL_CHECK_THROW(Result::ASSERTION_ERROR);
        }

        for (U64 i = 0; i < numberOfWorkers; i++) {
            workers.push_back(std::make_unique<T>(config));
        }
    }

    virtual ~Runner() = default;

    constexpr const T& getWorker(const U64& index = 0) const {
        return *workers[index];
    }

    constexpr const U64& getHead() const {
        return head;
    }

    Result applyToAllWorkers(const std::function<const Result(T&)>& modifier,
                             const bool block = false) {
        for (auto& worker : workers) {
             BL_CHECK(modifier(*worker));
        }

        if (block) {
            for (auto& worker : workers) {
                 BL_CHECK(worker->synchronize());
            }
        }

        return Result::SUCCESS;
    }

    bool enqueue(const std::function<const U64(T&)>& jobFunc) {
        // Return if there are no workers available.
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

    bool dequeue(U64* id) {
        // Return if there are no jobs.
        if (jobs.size() == 0) {
            return false;
        }

        const auto& job = jobs.front();

        // Synchronize front if all workers have jobs.
        if (jobs.size() == workers.size()) {
            job.worker->synchronize();
        }

        // Return if front isn't synchronized.
        if (!job.worker->isSynchronized()) {
            return false;
        }

        if (id != nullptr) {
            *id = job.id;
        }

        jobs.pop_front();

        return true;
    }

 private:
    struct Job {
        U64 id;
        std::unique_ptr<T>& worker;
    };

    U64 head = 0;
    std::deque<Job> jobs;
    std::vector<std::unique_ptr<T>> workers;
};

}  // namespace Blade

#endif
