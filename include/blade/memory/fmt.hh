#ifndef BLADE_MEMORY_FMT_HH
#define BLADE_MEMORY_FMT_HH

#include <array>

#include "blade/memory/custom.hh"

namespace fmt {

// TODO: Fix print.
template <>
struct formatter<Blade::ArrayShape> {
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx) {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto format(const Blade::ArrayShape& p, FormatContext& ctx) {
        return format_to(ctx.out(), "[A: {}, F: {}, T: {}, P: {}]", 
                         p.numberOfAspects(), p.numberOfFrequencyChannels(),
                         p.numberOfTimeSamples(), p.numberOfPolarizations());
    }
};

template <>
struct formatter<Blade::PhasorShape> {
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx) {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto format(const Blade::PhasorShape& p, FormatContext& ctx) {
        return format_to(ctx.out(), "[B: {}, A: {}, F: {}, T: {}, P: {}]", 
                         p.numberOfBeams(), p.numberOfAntennas(),
                         p.numberOfFrequencyChannels(), p.numberOfTimeSamples(), 
                         p.numberOfPolarizations());
    }
};

template <>
struct formatter<Blade::DelayShape> {
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx) {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto format(const Blade::DelayShape& p, FormatContext& ctx) {
        return format_to(ctx.out(), "[B: {}, A: {}]", 
                         p.numberOfBeams(), p.numberOfAntennas());
    }
};

} // namespace fmt

#endif
