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


if __name__ == "__main__":
    # Specify dimension of array.
    number_of_beams = 1
    number_of_antennas = 1
    number_of_frequency_channels = 2
    number_of_time_samples = 4
    number_of_polarizations = 1
    channelizer_rate = 2

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
    _a = np.linspace(0, mod.buffer_size() -1, mod.buffer_size(), dtype=np.int32)
    _b = np.linspace(0, mod.buffer_size() -1, mod.buffer_size(), dtype=np.int32)
    print(_a)
    _c = np.array(_a + _b * 1j).astype(np.complex64)
    input = _c.reshape((
            number_of_beams,
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
    for ibeam in range(number_of_beams):
        beam = input[ibeam]

        for iant in range(number_of_antennas):
            ant = beam[iant]

            for ichan in range(number_of_frequency_channels):
                ch_ant = ant[ichan]

                for ipol in range(number_of_polarizations):
                    pl_arr = ch_ant[:, ipol]

                    nspecs = pl_arr.size // channelizer_rate
                    arr_fft = np.zeros_like(pl_arr, dtype=np.complex64).reshape(nspecs, channelizer_rate)

                    for ispec in range(nspecs):
                        arr_fft[ispec] = pl_arr[ispec*channelizer_rate:(ispec+1)*channelizer_rate]

                    for i in range(channelizer_rate):
                        output[ibeam, iant, ichan, (i*nspecs):((i+1)*nspecs), ipol] = arr_fft[:,i]
    print(f"Channelize with Numpy took {time.time()-start:.2f} s.")

    print(np.array(bl_output, copy=False).real)
    print(output.flatten().real)

    # Check both answers.
    assert np.allclose(np.array(bl_output, copy=False), output.flatten(), rtol=0.01)
    print("Test successfully completed!")
