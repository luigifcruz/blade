#include <catch2/catch_session.hpp>

#include <jetstream/logger.hh>
#include <jetstream/plugin.hh>

extern "C" const JetstreamPluginAbi jetstream_plugin_abi;

int main(int argc, char* argv[]) {
    JST_LOG_SET_DEBUG_LEVEL(1);

    if (jetstream_plugin_abi.magic != JETSTREAM_PLUGIN_ABI_MAGIC) {
        return 1;
    }

    return Catch::Session().run(argc, argv);
}
