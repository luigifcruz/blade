import math
import time
import blade as bl
import numpy as np

class Test(bl.Pipeline):
    phasor: bl.Phasor
    frame_julian_date: float = (1649366473.0/ 86400.0) + 2440587.5
    frame_dut1: float = 0.0

    def __init__(self, phasor_config: bl.Phasor.Config):
        bl.Pipeline.__init__(self)
        _config = phasor_config
        _input = bl.Phasor.Input(self.input)
        self.phasor = self.connect(_config, _input)

    def phasor_size(self):
        return self.phasor.buffer_size()

    def run(self, frame_julian_date: float,
                  frame_dut1: float,
                  phasors: bl.vector.cpu.cf32):
        self.frame_julian_date = frame_julian_date
        self.frame_dut1 = frame_dut1
        self.copy(self.phasor.input(), input)
        self.compute()
        self.copy(phasors, self.phasor.phasors())
        self.synchronize()


if __name__ == "__main__":
    phasor_config = bl.Phasor.Config(
        number_of_beams = 1,
        number_of_antennas = 20,
        number_of_frequency_channels = 192,
        number_of_polarizations = 2,
        
        rf_frequency_hz = 0.0,
        channel_bandwidth_hz = 0.0,
        total_bandwidth_hz = 0.0,
        frequency_start_index = 0.0,

        reference_antenna_index = 0,
        array_reference_position = bl.LLA(
            LON = math.radians(-121.470733),
            LAT = math.radians(40.815987),
            ALT = 1020.86
        ),
        boresight_coordinate = bl.RA_DEC(
            RA = 0.94169,
            DEC = 1.079896295
        ),
        antenna_positions = ,
        antenna_calibrations = []
        beam_coordinates,

        block_size: 512
    )
    # Specify dimension of array.
    number_of_beams = 1
    number_of_antennas = 20
    number_of_frequency_channels = 96
    number_of_polarizations = 2

    reference_antenna_index,
    array_reference_position,
    boresight_coordinate,
    antenna_positions,
    antenna_calibrations, 
    beam_coordinates,


    # Initialize Blade pipeline.
    mod = Test(
        number_of_beams,
        number_of_antennas,
        number_of_frequency_channels,
        number_of_time_samples, 
        number_of_polarizations,
        channelizer_rate
    )

    # Generate test data with Python.
    _a = np.random.uniform(-int(2**16/2), int(2**16/2), mod.buffer_size())
    _b = np.random.uniform(-int(2**16/2), int(2**16/2), mod.buffer_size())
    _c = np.array(_a + _b * 1j).astype(np.complex64)
    input = _c.reshape((
            number_of_antennas, 
            number_of_frequency_channels,
            number_of_time_samples,
            number_of_polarizations,
        ))
    output = np.zeros_like(input, dtype=np.complex64)

    # Import test data from Python to Blade.
    bl_input = bl.vector.cpu.cf32(mod.buffer_size())
    bl_output = bl.vector.cpu.cf32(mod.buffer_size())

    np.copyto(np.array(bl_input, copy=False), input.flatten())
    np.copyto(np.array(bl_output, copy=False), output.flatten())

    # Channelize with Blade.
    start = time.time()
    mod.run(bl_input, bl_output)
    print(f"Channelize with Blade took {time.time()-start:.2f} s.")

    # Channelize with Numpy.
    start = time.time()
    for iant in range(number_of_antennas):
        ant = input[iant]

        for ichan in range(number_of_frequency_channels):
            ch_ant = ant[ichan]

            for ipol in range(number_of_polarizations):
                pl_arr = ch_ant[:, ipol]

                nspecs = pl_arr.size // channelizer_rate
                arr_fft = np.zeros_like(pl_arr, dtype=np.complex64).reshape(nspecs, channelizer_rate)

                for ispec in range(nspecs):
                    arr_fft[ispec] = np.fft.fft(pl_arr[ispec*channelizer_rate:(ispec+1)*channelizer_rate])

                for i in range(channelizer_rate):
                    output[iant, ichan, (i*nspecs):((i+1)*nspecs), ipol] = arr_fft[:,i]
    print(f"Channelize with Numpy took {time.time()-start:.2f} s.")

    # Check both answers.
    assert np.allclose(np.array(bl_output, copy=False), output.flatten(), rtol=0.01)
    print("Test successfully completed!")
