import math
import time
import blade as bl
import numpy as np
import pandas as pd

import astropy.constants as const
from astropy.coordinates import ITRS, SkyCoord
from astropy.time import Time
import astropy.units as u


class Test(bl.Pipeline):

    def __init__(self, _config: bl.Phasor.Config):
        bl.Pipeline.__init__(self)
        self.block_julian_date = bl.cpu.f64.Tensor((1,))
        self.block_dut1 = bl.cpu.f64.Tensor((1,))
        _input = bl.Phasor.Input(self.block_julian_date, self.block_dut1)
        self.phasor = self.connect(_config, _input)

    def output_shape(self):
        return self.phasor.phasors().shape()

    def run(self, block_julian_date: float,
                  block_dut1: float,
                  phasors: bl.cpu.cf32):
        self.block_julian_date[0] = block_julian_date
        self.block_dut1[0] = block_dut1
        self.compute()
        self.copy(phasors, self.phasor.phasors())
        self.synchronize()


if __name__ == "__main__":
    #
    # Blade Implementation
    #

    cal_shape = (20, 192, 1, 2)
    calibration = np.random.random(cal_shape) + 1j*np.random.random(cal_shape)
    calibration = np.array(calibration, dtype=np.complex128)
    bl_calibration = bl.cpu.cf64.ArrayTensor(cal_shape)
    np.copyto(bl_calibration.asnumpy(), calibration)

    phase_center_pos_rad = [0.64169, 1.079896295]
    beam1_pos_rad        = [0.63722, 1.07552424]
    beam2_pos_rad        = [0.65063, 1.08426835]

    phasor_config = bl.Phasor.Config(
        number_of_antennas = 20,
        number_of_frequency_channels = 192,
        number_of_polarizations = 2,
        
        observation_frequency_hz = 6500.125*1e6,
        channel_bandwidth_hz = 0.5e6,
        total_bandwidth_hz = 1.024e9,
        frequency_start_index = 352,

        reference_antenna_index = 0,
        array_reference_position = bl.LLA(
            LON = math.radians(-121.470733),
            LAT = math.radians(40.815987),
            ALT = 1020.86
        ),
        boresight_coordinate = bl.RA_DEC(
            RA = phase_center_pos_rad[0],
            DEC = phase_center_pos_rad[1]
        ),
        antenna_positions = [
            bl.XYZ(-2524041.5388905862, -4123587.965024342, 4147646.4222955606),    # 1c 
            bl.XYZ(-2524068.187873109, -4123558.735413135, 4147656.21282186),       # 1e 
            bl.XYZ(-2524087.2078100787, -4123532.397416349, 4147670.9866770394),    # 1g 
            bl.XYZ(-2524103.384010733, -4123511.111598937, 4147682.4133068994),     # 1h 
            bl.XYZ(-2524056.730228759, -4123515.287949227, 4147706.4850287656),     # 1k 
            bl.XYZ(-2523986.279601761, -4123497.427940991, 4147766.732988923),      # 2a 
            bl.XYZ(-2523970.301363642, -4123515.238502669, 4147758.790023165),      # 2b 
            bl.XYZ(-2523983.5419911123, -4123528.1422073604, 4147737.872218138),    # 2c 
            bl.XYZ(-2523941.5221860334, -4123568.125040547, 4147723.8292249846),    # 2e 
            bl.XYZ(-2524074.096220788, -4123468.5182652213, 4147742.0422435375),    # 2h 
            bl.XYZ(-2524058.6409591637, -4123466.5112451194, 4147753.4513993543),   # 2j 
            bl.XYZ(-2524026.989692545, -4123480.9405167866, 4147758.2356800516),    # 2l 
            bl.XYZ(-2524048.5254066754, -4123468.3463909747, 4147757.835369889),    # 2k 
            bl.XYZ(-2524000.5641107005, -4123498.2984570004, 4147756.815976133),    # 2m 
            bl.XYZ(-2523945.086670364, -4123480.3638816103, 4147808.127865142),     # 3d 
            bl.XYZ(-2523950.6822576034, -4123444.7023326857, 4147839.7474427638),   # 3l 
            bl.XYZ(-2523880.869769226, -4123514.3375464156, 4147813.413426994),     # 4e 
            bl.XYZ(-2523930.3747946257, -4123454.3080821196, 4147842.6449955846),   # 4g 
            bl.XYZ(-2523898.1150373477, -4123456.314794732, 4147860.3045849088),    # 4j 
            bl.XYZ(-2523824.598229116, -4123527.93080514, 4147833.98936114),        # 5b
        ],
        antenna_calibrations = bl_calibration,
        beam_coordinates = [
            bl.RA_DEC(*beam1_pos_rad),
            bl.RA_DEC(*beam2_pos_rad)
        ],

        block_size = 512
    )

    mod = Test(phasor_config)

    bl_output = bl.cpu.cf32.PhasorTensor(mod.output_shape())

    mod.run((1649366473.0/ 86400) + 2440587.5, 0.0, bl_output)

    bl_phasors = bl_output.asnumpy().reshape((2, 20, 192, 2))
    bl_phasors = bl_phasors[:, :, :, 0]

    #
    # Python Implementation
    #

    REFANT = "1C"
    ITRF = pd.DataFrame.from_dict({'x': {'1A': -2524036.0307912203, '1B': -2524012.4958892134, '1C': -2524041.5388905862, '1D': -2524059.792573754, '1E': -2524068.187873109, '1F': -2524093.716340507, '1G': -2524087.2078100787, '1H': -2524103.384010733, '1J': -2524096.980570317, '1K': -2524056.730228759, '2A': -2523986.279601761, '2B': -2523970.301363642, '2C': -2523983.5419911123, '2D': -2523957.2473999085, '2E': -2523941.5221860334, '2F': -2524059.106099003, '2G': -2524084.5774130877, '2H': -2524074.096220788, '2J': -2524058.6409591637, '2K': -2524048.5254066754, '2L': -2524026.989692545, '2M': -2524000.5641107005, '3C': -2523913.265779332, '3D': -2523945.086670364, '3E': -2523971.439791098, '3F': -2523989.1620624275, '3G': -2523998.124952975, '3H': -2523992.8464242537, '3J': -2523932.8971014554, '3L': -2523950.6822576034, '4E': -2523880.869769226, '4F': -2523913.382185706, '4G': -2523930.3747946257, '4H': -2523940.4924399494, '4J': -2523898.1150373477, '4K': -2523896.075066118, '4L': -2523886.8518362255, '5B': -2523824.598229116, '5C': -2523825.0029098387, '5E': -2523818.829627262, '5G': -2523843.4903899855, '5H': -2523836.636021752}, 'y': {'1A': -4123528.101172219, '1B': -4123555.1766721113, '1C': -4123587.965024342, '1D': -4123572.448681901, '1E': -4123558.735413135, '1F': -4123544.129301442, '1G': -4123532.397416349, '1H': -4123511.111598937, '1J': -4123480.786854586, '1K': -4123515.287949227, '2A': -4123497.427940991, '2B': -4123515.238502669, '2C': -4123528.1422073604, '2D': -4123560.6492725424, '2E': -4123568.125040547, '2F': -4123485.615619698, '2G': -4123471.5634082295, '2H': -4123468.5182652213, '2J': -4123466.5112451194, '2K': -4123468.3463909747, '2L': -4123480.9405167866, '2M': -4123498.2984570004, '3C': -4123517.062782675, '3D': -4123480.3638816103, '3E': -4123472.6180766555, '3F': -4123471.266121543, '3G': -4123467.6450268496, '3H': -4123464.322817822, '3J': -4123470.89044144, '3L': -4123444.7023326857, '4E': -4123514.3375464156, '4F': -4123479.7060887963, '4G': -4123454.3080821196, '4H': -4123445.672260385, '4J': -4123456.314794732, '4K': -4123477.339537938, '4L': -4123483.024180943, '5B': -4123527.93080514, '5C': -4123540.8439693213, '5E': -4123551.1077656075, '5G': -4123539.9993453375, '5H': -4123534.166729928}, 'z': {'1A': 4147706.408318585, '1B': 4147693.5484577166, '1C': 4147646.4222955606, '1D': 4147648.2406134596, '1E': 4147656.21282186, '1F': 4147655.6831260016, '1G': 4147670.9866770394, '1H': 4147682.4133068994, '1J': 4147716.343429463, '1K': 4147706.4850287656, '2A': 4147766.732988923, '2B': 4147758.790023165, '2C': 4147737.872218138, '2D': 4147721.727385693, '2E': 4147723.8292249846, '2F': 4147734.198131659, '2G': 4147732.8048639772, '2H': 4147742.0422435375, '2J': 4147753.4513993543, '2K': 4147757.835369889, '2L': 4147758.2356800516, '2M': 4147756.815976133, '3C': 4147791.1821111576, '3D': 4147808.127865142, '3E': 4147799.766493265, '3F': 4147790.551974626, '3G': 4147788.711874468, '3H': 4147796.3664053706, '3J': 4147824.7980390238, '3L': 4147839.7474427638, '4E': 4147813.413426994, '4F': 4147827.8364580916, '4G': 4147842.6449955846, '4H': 4147844.9899415076, '4J': 4147860.3045849088, '4K': 4147840.538272895, '4L': 4147840.260497064, '5B': 4147833.98936114, '5C': 4147821.048239712, '5E': 4147814.906896719, '5G': 4147810.7145798537, '5H': 4147820.5849095713}})

    def compute_uvw(ts, source, ant_coordinates, ref_coordinates):
        """Computes UVW antenna coordinates with respect to reference

        Copyright 2021 Daniel Estevez <daniel@destevez.net>

        Args:
            ts: array of Times to compute the coordinates
            source: source SkyCoord
            ant_coordinates: antenna ECEF coordinates.
                This is indexed as (antenna_number, xyz)
            ref_coordinates: phasing reference centre coordinates.
                This is indexed as (xyz)

        Returns:
            The UVW coordinates in metres of each of the baselines formed
            between each of the antennas and the phasing reference. This
            is indexed as (time, antenna_number, uvw)
        """
        baselines_itrs = ant_coordinates - ref_coordinates

        # Calculation of vector orthogonal to line-of-sight
        # and pointing due north.
        north_radec = [source.ra.deg, source.dec.deg + 90]
        if north_radec[1] > 90:
            north_radec[1] = 180 - north_radec[1]
            north_radec[0] = 180 + north_radec[0]
        north = SkyCoord(ra=north_radec[0]*u.deg, dec=north_radec[1]*u.deg)

        source_itrs = source.transform_to(ITRS(obstime=Time(ts))).cartesian
        north_itrs = north.transform_to(ITRS(obstime=Time(ts))).cartesian
        east_itrs = north_itrs.cross(source_itrs)

        ww = baselines_itrs @ source_itrs.xyz.value
        vv = baselines_itrs @ north_itrs.xyz.value
        uu = baselines_itrs @ east_itrs.xyz.value
        uvw = np.stack((uu.T, vv.T, ww.T), axis=-1)

        return uvw


    def create_delay_phasors(delay, startchan, nchans, chanwidth):
        # IF frequency index, startchan -> startchan + number of freq chan
        freqs_idx = np.arange(startchan, startchan + nchans) 

        # Now convert to Hz
        freqs = freqs_idx * chanwidth

        return np.exp(-1j*2*np.pi*delay*freqs)


    def get_fringe_rate(delay, rffreq, totalbw):
        return np.exp(-1j*2*np.pi*delay * (rffreq - (totalbw/2.)))


    # Observation parameters
    rfFrequencyHz = 6500.125*1e6
    channelBandwidthHz = 0.5e6
    totalBandwidthHz = 1.024e9
    frequencyStartIndex = 352
    obsnchan = 192 

    boresight_ra, boresight_dec = np.rad2deg(phase_center_pos_rad)

    source = SkyCoord(boresight_ra, boresight_dec, unit='deg')

    beam1_ra, beam1_dec = np.rad2deg(beam1_pos_rad)
    beam2_ra, beam2_dec = np.rad2deg(beam2_pos_rad)
    beam1 = SkyCoord(beam1_ra, beam1_dec, unit='deg')
    beam2 = SkyCoord(beam2_ra, beam2_dec, unit='deg')

    antnames = ["1C", "1E", "1G", "1H", "1K", "2A", "2B", "2C", "2E", "2H", "2J", "2L", "2K", "2M", "3D", "3L", "4E", "4G", "4J", "5B"]
    itrf_sub = ITRF.loc[antnames]
    irefant = itrf_sub.index.values.tolist().index(REFANT)

    #spectra_n = (hdr['PKTSTART'] + hdr['PKTSTOP']) / 2. 
    #t = hdr['SYNCTIME'] + spectra_n * hdr['TBIN']

    t = 1649366473 #unix time
    ts = Time(t, format='unix')

    # These UVWs are centred at the reference antenna, 
    # i.e. UVW_irefant = [0, 0, 0]
    boresight_uvw = compute_uvw(ts, source, 
            itrf_sub[['x','y','z']], itrf_sub[['x','y','z']].values[irefant])

    beam1_uvw = compute_uvw(ts, beam1,
            itrf_sub[['x','y','z']], itrf_sub[['x','y','z']].values[irefant])

    beam2_uvw = compute_uvw(ts, beam2,
            itrf_sub[['x','y','z']], itrf_sub[['x','y','z']].values[irefant])

    # subtracting the W-coordinate from boresights
    # to get the delays needed to steer relative to boresight
    delays1 = (beam1_uvw[:,2] - boresight_uvw[:,2] ) / const.c.value
    delays2 = (beam2_uvw[:,2] - boresight_uvw[:,2] ) / const.c.value

    IF_freq = np.arange(frequencyStartIndex, 
            frequencyStartIndex+obsnchan)

    py_phasors = np.zeros((2, 20, 192), dtype=np.complex128)
    for i in range(len(delays1)):
        delay1 = delays1[i]
        delay2 = delays2[i]

        py_phasors[0, i, :] = create_delay_phasors(delay1, frequencyStartIndex,
                obsnchan, channelBandwidthHz)
        py_phasors[0, i, :] *= get_fringe_rate(delay1, rfFrequencyHz,
                totalBandwidthHz)
        py_phasors[0, i, :] *= calibration[i, :, 0, 0]

        py_phasors[1, i, :] = create_delay_phasors(delay2, frequencyStartIndex,
                obsnchan, channelBandwidthHz)
        py_phasors[1, i, :] *= get_fringe_rate(delay2, rfFrequencyHz,
                totalBandwidthHz)
        py_phasors[1, i, :] *= calibration[i, :, 0, 0]

    #
    # Compare Results
    #

    assert np.allclose(py_phasors, bl_phasors, rtol=0.01)
    print("Test successfully completed!")
