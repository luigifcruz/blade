import sys
import time
import blade as bl
import numpy as np

class Test(bl.Pipeline):
    channelizer: bl.Channelizer
    input = bl.vector.cuda.cf32()

    def __init__(self,
                 number_of_beams,
                 number_of_antennas,
                 number_of_frequency_channels,
                 number_of_time_samples,
                 number_of_polarizations,
                 channelizer_rate):
        bl.Pipeline.__init__(self)
        _config = bl.Channelizer.Config(
            number_of_beams,
            number_of_antennas,
            number_of_frequency_channels,
            number_of_time_samples,
            number_of_polarizations,
            channelizer_rate,
            512
        )
        _input = bl.Channelizer.Input(self.input)
        self.channelizer = self.connect(_config, _input)

    def buffer_size(self):
        return self.channelizer.buffer_size()

    def run(self, input: bl.vector.cpu.cf32,
                  output: bl.vector.cpu.cf32):
        self.copy(self.channelizer.input(), input)
        self.compute()
        self.copy(output, self.channelizer.output())
        self.synchronize()

def trial(number_of_beams, number_of_antennas, number_of_frequency_channels, 
        number_of_time_samples, number_of_polarizations, channelizer_rate):
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
            number_of_beams * number_of_antennas,
            number_of_frequency_channels,
            number_of_time_samples,
            number_of_polarizations,
        ))

    # Compute the FFT sizes.
    nspecs = number_of_time_samples // channelizer_rate
    number_of_new_time_samples = nspecs
    number_of_new_frequency_channels = number_of_frequency_channels * channelizer_rate

    # Define output buffer.
    _a = np.random.uniform(-int(2**16/2), int(2**16/2), mod.buffer_size())
    _b = np.random.uniform(-int(2**16/2), int(2**16/2), mod.buffer_size())
    _c = np.array(_a + _b * 1j).astype(np.complex64)
    output = _c.reshape((
            number_of_beams * number_of_antennas,
            number_of_new_frequency_channels,
            number_of_new_time_samples,
            number_of_polarizations,
        ))

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
    for ibeam in range(number_of_beams * number_of_antennas):
        beam = input[ibeam]

        for ichan in range(number_of_frequency_channels):
            time_pol = beam[ichan]

            for ipol in range(number_of_polarizations):
                time_arr = time_pol[:, ipol]

                for ispec in range(nspecs):
                    output[ibeam,
                            ichan*channelizer_rate :
                            (ichan+1)*channelizer_rate,
                            ispec, ipol] =\
                            np.fft.fftshift(np.fft.fft(time_arr[ispec*channelizer_rate:(ispec+1)*channelizer_rate]))
    print(f"Channelize with Numpy took {time.time()-start:.2f} s.")

    # Check both answers.
    assert np.allclose(np.array(bl_output, copy=False), output.flatten(), rtol=0.01)
    print("Test successfully completed!")


if __name__ == "__main__":
    trial(int(sys.argv[1]),
          int(sys.argv[2]),
          int(sys.argv[3]), 
          int(sys.argv[4]), 
          int(sys.argv[5]), 
          int(sys.argv[6])) 
