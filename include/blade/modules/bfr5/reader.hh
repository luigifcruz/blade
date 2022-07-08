#ifndef BLADE_MODULES_BFR5_READER_HH
#define BLADE_MODULES_BFR5_READER_HH

#include <filesystem>
#include <string>

#include "blade/base.hh"
#include "blade/module.hh"

extern "C" {
#include "bfr5.h"
}

namespace Blade::Modules::Bfr5 {

class BLADE_API Reader {
 public:
    explicit Reader(const std::string& filepath);

 private:
    BFR5_file_t bfr5_file;
};

}  // namespace Blade::Modules

#endif

