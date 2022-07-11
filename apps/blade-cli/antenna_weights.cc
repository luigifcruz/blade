#include <complex.h>
#include <bfr5.h>
#include <vector>

#define CAST_COMPLEX_DOUBLE(complex_float) \
    std::complex<double>((double)complex_float.re, (double)complex_float.im)

void gather_antenna_weights_from_bfr5_cal(
	const complex_float_t* cal, // [NCHAN=slowest, NPOL, NANTS=fastest)]
    const uint32_t nants,
    const uint32_t nchan,
    const uint32_t npol,
	const uint32_t starting_channel, // the first channel
	const uint32_t number_of_channels, // the number of channels
	std::complex<double>* weights // [NANTS=slowest, number_of_channels, NPOL=fastest)]
) {
    const size_t cal_ant_stride = 1;
    const size_t cal_pol_stride = nants*cal_ant_stride;
    const size_t cal_chan_stride = npol*cal_pol_stride;

    const size_t weights_pol_stride = 1;
    const size_t weights_chan_stride = npol*weights_pol_stride;
    const size_t weights_ant_stride = number_of_channels*weights_chan_stride;

    size_t ant_i, chan_i, pol_i;

    for(ant_i = 0; ant_i < nants; ant_i++) {
        for(chan_i = 0; chan_i < number_of_channels; chan_i++) {
            for(pol_i = 0; pol_i < npol; pol_i++) {
                weights[
                    ant_i * weights_ant_stride +
                    chan_i * weights_chan_stride +
                    pol_i * weights_pol_stride
                ] = CAST_COMPLEX_DOUBLE(cal[
                    (starting_channel + chan_i) * cal_chan_stride +
                    pol_i * cal_pol_stride +
                    ant_i * cal_ant_stride
                ]);
            }
        }
    }

}