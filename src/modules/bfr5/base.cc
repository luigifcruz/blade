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
        this->getDiminfo_nants(),
        this->getDiminfo_nchan(),
        this->getDiminfo_ntimes(),
        this->getDiminfo_npol(),
        this->getDiminfo_nbeams()
    );

    BFR5close(&this->bfr5_file);
}

std::vector<RA_DEC> Reader::getBeaminfo_coordinates() {
    std::vector<RA_DEC> coordinates;
    coordinates.reserve(this->bfr5_file.beam_info.ra_elements);
    for (size_t i = 0; i < coordinates.capacity(); i++) {
        coordinates.push_back({
            this->bfr5_file.beam_info.ras[i],
            this->bfr5_file.beam_info.decs[i]
        });
    }
    return coordinates;
}

std::vector<XYZ> Reader::getTelinfo_antenna_positions() {
    std::vector<XYZ> pos;
    pos.reserve(this->bfr5_file.tel_info.antenna_position_elements/3);
    double* ant_pos = this->bfr5_file.tel_info.antenna_positions;

    std::string ant_frame = std::string(this->bfr5_file.tel_info.antenna_position_frame);
    if((ant_frame.compare("xyz") * ant_frame.compare("XYZ")) != 0) {
        ant_pos = (double*) malloc(pos.capacity()*sizeof(double));
        memcpy(ant_pos, this->bfr5_file.tel_info.antenna_positions, pos.capacity()*sizeof(double));

        if((ant_frame.compare("ecef") * ant_frame.compare("ECEF")) == 0) {
            calc_position_to_xyz_frame_from_ecef(
                ant_pos,
                pos.capacity(),
                this->bfr5_file.tel_info.latitude,
                this->bfr5_file.tel_info.longitude,
                this->bfr5_file.tel_info.altitude
            );
        }
        else if((ant_frame.compare("enu") * ant_frame.compare("ENU")) == 0) {
            calc_position_to_xyz_frame_from_enu(
                ant_pos,
                pos.capacity(),
                this->bfr5_file.tel_info.latitude,
                this->bfr5_file.tel_info.longitude,
                this->bfr5_file.tel_info.altitude
            );
        }
    }

    for(size_t i = 0; i < pos.capacity(); i++) {
        pos.push_back({
            ant_pos[3*i+0],
            ant_pos[3*i+1],
            ant_pos[3*i+2],
        });
    }

    if((ant_frame.compare("xyz") | ant_frame.compare("XYZ")) != 0) {
        free(ant_pos);
    }
    return pos;
}

}  // namespace Blade::Modules::Bfr5
