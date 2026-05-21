#include <jetstream/plugin.hh>
#include <jetstream/registry.hh>

JST_PLUGIN_ABI(
    "blade",
    "2.0.0",
    JETSTREAM_VERSION_CURRENT,
    static_cast<uint64_t>(Jetstream::DeviceType::CUDA),
    static_cast<uint64_t>(Jetstream::RuntimeType::NATIVE)
)
