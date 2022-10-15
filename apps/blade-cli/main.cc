#include "blade-cli/base.hh"

// Bootstrap Blade::CLI instance.
int main(int argc, char **argv) {
    if (Blade::CLI::Start(argc, argv) != Blade::Result::SUCCESS) {
        return 1;
    }
    return 0;
}
