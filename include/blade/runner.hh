#ifndef BLADE_RUNNER_HH
#define BLADE_RUNNER_HH

#include <deque>
#include <vector>
#include <memory>

#include "blade/logger.hh"
#include "blade/module.hh"
#include "blade/pipeline.hh"
#include "blade/macros.hh"

namespace Blade {

template<class Pipeline>
class BLADE_API Runner {
 public:
    static std::unique_ptr<Runner<Pipeline>> New(const U64& numberOfWorkers,
                                                 const typename Pipeline::Config& config,
                                                 const BOOL& printET = true) {
        return std::make_unique<Runner<Pipeline>>(numberOfWorkers, config, printET);
    }

    explicit Runner(const U64& numberOfWorkers,
                    const typename Pipeline::Config& config,
                    const BOOL& printET = true) {
        if (printET) {
            BL_LOG_PRINT_ET();
        }

        BL_INFO("Instantiating new runner.");

        if (numberOfWorkers == 0) {
            BL_FATAL("Number of worker has to be larger than zero.");
            BL_CHECK_THROW(Result::ASSERTION_ERROR);
        }

        for (U64 i = 0; i < numberOfWorkers; i++) {
            BL_DEBUG("Initializing new worker.");
            workers.push_back(std::make_unique<Pipeline>(config));
        }
    }

    constexpr Pipeline& getWorker(const U64& index = 0) const {
        return *workers[index];
    }

    constexpr const U64& getHead() const {
        return head;
    }
    
    constexpr const bool slotAvailable() const {
        return jobs.size() != workers.size();
    }

    constexpr const bool empty() const {
        return jobs.size() == 0;
    }
    
    constexpr Pipeline& getNextWorker() {
        return *workers[head];
    }

    const Result applyToAllWorkers(const std::function<const Result(Pipeline&)>& modifier,
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

    bool enqueue(const std::function<const U64(Pipeline&)>& jobFunc) {
        // Return if there are no workers available.
        if (jobs.size() == workers.size()) {
            return false;
        }

        try {
            jobs.push_back({
                .id = jobFunc(*workers[head]),
                .worker = workers[head],
                .workerId = head,
            });
        } catch (const Result& err) {
            // Print user friendly error and issue fatal error.
            if (err == Result::PLAN_ERROR_ACCUMULATION_COMPLETE) {
                BL_FATAL("Can't accumulate block because buffer is full.");
                BL_CHECK_THROW(err);
            }

            if (err == Result::PLAN_ERROR_DESTINATION_NOT_SYNCHRONIZED) {
                BL_FATAL("Can't transfer data because destination is not synchronized.");
                BL_CHECK_THROW(err);
            }

            if (err == Result::PLAN_ERROR_NO_ACCUMULATOR) {
                BL_FATAL("This mode doesn't support accumulation.");
                BL_CHECK_THROW(err);
            }

            if (err == Result::PLAN_ERROR_NO_SLOT) {
                BL_FATAL("No slot available after compute. Data has nowhere to go.")
                BL_CHECK_THROW(err);
            }

            // Ignore if throw was a skip operation.
            if (err == Result::PLAN_SKIP_ACCUMULATION_INCOMPLETE || 
                err == Result::PLAN_SKIP_COMPUTE_INCOMPLETE ||
                err == Result::PLAN_SKIP_USER_INITIATED ||
                err == Result::PLAN_SKIP_NO_DEQUEUE || 
                err == Result::PLAN_SKIP_NO_SLOT) {
                return false;
            }

            // Ignore if throw originates from exhaustion.
            if (err == Result::EXHAUSTED) {
                return false;
            }

            BL_FATAL("Unknown error.");

            // Fatal error otherwise.
            BL_CHECK_THROW(err);
        }

        // Bump job queue head index.
        head = (head + 1) % workers.size();

        return true;
    }

    bool dequeue(U64* id, U64* workerId = nullptr) {
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

        if (workerId != nullptr) {
            *workerId = job.workerId;
        }

        jobs.pop_front();

        return true;
    }

 private:
    struct Job {
        U64 id;
        std::unique_ptr<Pipeline>& worker;
        U64 workerId;
    };

    U64 head = 0;
    std::deque<Job> jobs;
    std::vector<std::unique_ptr<Pipeline>> workers;
};

}  // namespace Blade

#endif
