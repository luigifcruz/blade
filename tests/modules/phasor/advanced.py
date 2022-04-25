import math
import time
import blade as bl
import numpy as np

class Test(bl.Pipeline):
    phasor: bl.Phasor
    frame_julian_date = bl.vector.cpu.f64(1)
    frame_dut1 = bl.vector.cpu.f64(1)

    def __init__(self, phasor_config: bl.Phasor.Config):
        bl.Pipeline.__init__(self)
        _config = phasor_config
        _input = bl.Phasor.Input(self.frame_julian_date, self.frame_dut1)
        self.phasor = self.connect(_config, _input)

    def phasors_size(self):
        return self.phasor.phasors_size()

    def run(self, frame_julian_date: float,
                  frame_dut1: float,
                  phasors: bl.vector.cpu.cf32):
        self.frame_julian_date[0] = frame_julian_date
        self.frame_dut1[0] = frame_dut1
        self.compute()
        self.copy(phasors, self.phasor.phasors())
        self.synchronize()


if __name__ == "__main__":
    phasor_config = bl.Phasor.Config(
        number_of_beams = 1,
        number_of_antennas = 20,
        number_of_frequency_channels = 192,
        number_of_polarizations = 2,
        
        rf_frequency_hz = 6500.125*1e6,
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
            RA = 0.64169,
            DEC = 1.079896295
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
        antenna_calibrations = np.zeros(20*192*2),
        beam_coordinates = [
            bl.RA_DEC(0.63722, 1.07552424)
        ],

        block_size = 512
    )

    mod = Test(phasor_config)

    bl_phasors = bl.vector.cpu.cf32(mod.phasors_size())

    mod.run((1649366473.0/ 86400) + 2440587.5, 0.0, bl_phasors)

    np_arr = np.array(bl_phasors, copy=False)
    np_arr = np_arr.reshape((1, 20, 192, 2))

    print(np_arr.dtype)

    import matplotlib.pyplot as plt

    IF_freq = np.arange(352, 352+192) 

    for phasor in np_arr[0,:,:,:]:
        print(phasor[0])
        plt.plot(IF_freq, np.angle(phasor), ".")

    plt.savefig('foo.pdf')
