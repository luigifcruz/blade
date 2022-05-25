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

template<class T>
class BLADE_API Runner {
 public:
    static std::unique_ptr<Runner<T>> New(const U64& numberOfWorkers,
                                          const typename T::Config& config,
                                          const BOOL& printET = true) {
        return std::make_unique<Runner<T>>(numberOfWorkers, config, printET);
    }

    explicit Runner(const U64& numberOfWorkers,
                    const typename T::Config& config,
                    const BOOL& printET = true) {
        if (printET) {
            BL_INFO(R"(

Welcome to BLADE (Breakthrough Listen Accelerated DSP Engine)!
Version {} | Build Type: {}
                .-.
    .-""`""-.    |(0 0)
_/`oOoOoOoOo`\_ \ \-/
'.-=-=-=-=-=-=-.' \/ \
`-=.=-.-=.=-'    \ /\
    ^  ^  ^       _H_ \
            )", BLADE_VERSION_STR, BLADE_BUILD_TYPE);
        }

        BL_INFO("Instantiating new runner.");

        if (numberOfWorkers == 0) {
            BL_FATAL("Number of worker has to be larger than zero.");
            BL_CHECK_THROW(Result::ASSERTION_ERROR);
        }

        for (U64 i = 0; i < numberOfWorkers; i++) {
            BL_DEBUG("Initializing new worker.");
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
    
    constexpr const bool slotAvailable() const {
        return jobs.size() != workers.size();
    }
    
    constexpr const T& getNextWorker() const {
        return *workers[head];
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

/*
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&#G#&&&&&&&&&&&&&&&
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&GJ5#&&&&&&&&&&&&&&&&
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&@&P!?B&&&&&&&&&&&&&&&&&&
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&@&&#Y^~G&&&&&&&&&&&&&&&&&&&&
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&#GYJ&G75&&&&&&&&&&&&&&&&&&&&&&
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&#7:.  ?5^Y#&@&&&&&&&&&&&&&&&&&&&&
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&@G^    .#!  .!5#@&&&&&&&&&&&&&&&&&&
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&Y.     ?#      :?G&&&&&&&&&&&&&&&&&
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&#7       PB^.       ~#&&&&&&&&&&&&&&&
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&G^        P@&B57:     5@&&&&&&&&&&&&&&
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&BGBBB##.         5@&&@@#GJ~. J@&&&&&&&&&&&&&&
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&P.   ^&~         Y@&&&&&&@&#55&&&&&&&&&&&&&&&
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&@5    YG         J@&&&&&&&&&@&&&&&&&&&&&&&&&&
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&@J   :#^        ?@&&&&&&&&&&&&&&&&&&&&&&&&&&
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&@&GB&&&&&#P#@@?   J5        7@&&&&&&&&&&&&&&&&&&&&&&&&&&
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&@@@@@&&&&&&&@@#Y^  7&B?BY :?G&!  .B:       !@&&&&&&&&&&&&&&&&&&&&&&&&&&
&&&&&&&&&&&&&&&&&&&&&&&&&@&GP5YJ??BP^:~YBP!. .~JP7 ~&^   7##J~.?J       ~&&&&&&&&&&&&&&&&&&&&&&&&&&&
&&&&&&&&&&&&&&&&&&&&&&&@&5^     :57 :?5?^:~7?GG!   G5  ^PY~#7?5P#^      ^&&&&&&&&&&&&&&&&&&&&&&&&&&&
&&&&&&&&&&&&&&&&&&&&&@#Y^      ?5~~YY?!777~..7Y7^ ^&: JG! ~#.  :!Y5J!:  :&&&&&&&&&&&&&&&&&&&&&&&&&&&
&&&&&&&&&&&&&&&&&&&@#J:      :55JPPJ7!^.       ^?JG5!GJ   !B      .^?Y5?!&&&&&&&&&&&&&&&&&&&&&&&&&&&
&&&&&&&&&&&&&&&&&@B?.       7#BGY!:              :#B5:    ?G          :7G&&&&&&&&&&&&&&&&&&&&&&&&&&&
&&&&&&&&&&&&&&&@B?.        7B&B5J??777!!!~~~^^:: ^#?      YP       ~J5G#&&&&&&&&&&&&&&&&&&&&&&&&&&&&
&&&&&&&&&&&&&&G7.           .:^!?JJYJJ?77!!!777?YJ?J?~    55       !@@&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
&&&&&&&&&&@&P!       :^!J5:        .:^~7??JJ!~:..:!?Y5P?: GJ        Y@&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
&&&&&&&&&&5^  .^!?YPB#&@@&.              ?&@@&&#BPY??PGJJY#?        .B&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
&&&&&&&&G?!J5G#&&@@&&&&&&B             :P@&&&&&&&&@@&&G?!YB~         !&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
&&&&&&&&#&@@@&&&&&&&&&&&&B^:.         !#@&&&&&&&&&&&&&&B?:            5@&&&&&&&&&&&&&&&&&&&&&&&&&&&&
&&&&&&&&&&&&&&&&&&&&&&&&&5!?JJJ?7!^..Y&&&&&&&&&&&&&&&&&&@#Y:          .#&&&&&&&&&&&&&&&&&&&&&&&&&&&&
&&&&&&&&&&&&&&&&&&&&&&&&&~    .:^!P##&&&&&&&&&&&&&&&&&&&&&@&5^         !&&&&&&&&&&&&&&&&&&&&&&&&&&&&
&&&&&&&&&&&&&&&&&&&&&&&&#.      .5&&&&&&&&&&&&&&&&&&&&&&&&&&@&P~        5@&&&&&&&&&&&&&&&&&&&&&&&&&&
&&&&&&&&&&&&&&&&&&&&&&&@5      7#@&&&&&&&&&&&&&&&&&&&&&&&&&&&&@&G!      :#&&&&&&&&&&&&&&&&&&&&&&&&&&
&&&&&&&&&&&&&&&&&&&&&&&@7    :P@&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&B7.    7@&&&&&&&&&&&&&&&&&&&&&&&&&
&&&&&&&&&&&&&&&&&&&&&&&&:   ?#@&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&@BJ:   5@&&&&&&&&&&&&&&&&&&&&&&&&
&&&&&&&&&&&&&&&&&&&&&&&G  ^G@&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&@#Y: .#&&&&&&&&&&&&&&&&&&&&&&&&
&&&&&&&&&&&&&&&&&&&&&&@J J&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&@&5^7&&&&&&&&&&&&&&&&&&&&&&&&
&&&&&&&&&&&&&&&&&&&&&&&JG@&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&@&P#&&&&&&&&&&&&&&&&&&&&&&&
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
*/
