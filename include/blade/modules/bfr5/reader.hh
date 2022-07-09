#ifndef BLADE_MODULES_BFR5_READER_HH
#define BLADE_MODULES_BFR5_READER_HH

#include <filesystem>
#include <string>

#include "blade/base.hh"
#include "blade/module.hh"

extern "C" {
#include "bfr5.h"
#include "radiointerferometryc99.h"
}

namespace Blade::Modules::Bfr5 {

class BLADE_API Reader {
public:
   explicit Reader(const std::string& filepath);
   
   std::vector<RA_DEC> getBeaminfo_coordinates();

   constexpr const U64 getDiminfo_nants() const {
      return this->bfr5_file.dim_info.nants;
   }

   constexpr const U64 getDiminfo_nbeams() const {
      return this->bfr5_file.dim_info.nbeams;
   }

   constexpr const U64 getDiminfo_nchan() const {
      return this->bfr5_file.dim_info.nchan;
   }

   constexpr const U64 getDiminfo_npol() const {
      return this->bfr5_file.dim_info.npol;
   }

   constexpr const U64 getDiminfo_ntimes() const {
      return this->bfr5_file.dim_info.ntimes;
   }

   constexpr const LLA getTelinfo_lla() const {
      return LLA(
         this->bfr5_file.tel_info.latitude,
         this->bfr5_file.tel_info.longitude,
         this->bfr5_file.tel_info.altitude
      );
   }

   std::vector<XYZ> getTelinfo_antenna_positions();

   constexpr const RA_DEC getObsinfo_phase_center() const {
      return RA_DEC(
         this->bfr5_file.obs_info.phase_center_ra,
         this->bfr5_file.obs_info.phase_center_dec
      );
   }


 private:
    BFR5_file_t bfr5_file;
};

}  // namespace Blade::Modules

#endif

