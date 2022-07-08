#include "blade/modules/bfr5/reader.hh"

namespace Blade::Modules::Bfr5 {

Reader::Reader(const std::string& filepath) {
    BL_INFO("===== BFR5 Reader Module Configuration");

    if (!std::filesystem::exists(filepath)) {
        BL_FATAL("Input file ({}) doesn't not exist.", filepath);
    }

    BFR5open(filepath.c_str(), &this->bfr5_file);
    BFR5read_all(&this->bfr5_file);

    BL_INFO("Input File Path: {}", filepath);
    BL_INFO("Dim_info: [{}, {}, {}, {}] - {} Beams",
        this->bfr5_file.dim_info.nants,
        this->bfr5_file.dim_info.nchan,
        this->bfr5_file.dim_info.ntimes,
        this->bfr5_file.dim_info.npol,
        this->bfr5_file.dim_info.nbeams
    );

    BFR5close(&this->bfr5_file);
}

}  // namespace Blade::Modules::Bfr5
