import numpy as np
import blade as bl

@bl.runner
class Pipeline:
    def __init__(self, input_shape, phasor_shape, output_shape, config):
        self.input.buf = bl.array_tensor(input_shape, dtype=bl.cf32)
        self.input.phasors = bl.phasor_tensor(phasor_shape, dtype=bl.cf32)
        self.output.buf = bl.array_tensor(output_shape, dtype=bl.cf32)

        input = (self.input.buf, self.input.phasors)
        self.module.beamformer = bl.module(bl.beamformer, config, input, telescope=bl.ata)

    def transfer_in(self, buf, phasors):
        self.copy(self.input.buf, buf)
        self.copy(self.input.phasors, phasors)

    def transfer_out(self, buf):
        self.copy(self.output.buf, self.module.beamformer.get_output())
        self.copy(buf, self.output.buf)


if __name__ == "__main__":
    # Specify dimension of array.
    input_shape = (2, 192, 512, 2)
    phasor_shape = (1, 2, 192, 1, 2)
    output_shape = (2, 192, 512, 2)

    config = {
        'enable_incoherent_beam': True,
        'enable_incoherent_beam_sqrt': True,
    }

    host_input = bl.array_tensor(input_shape, dtype=bl.cf32, device=bl.cpu)
    host_phasors = bl.phasor_tensor(phasor_shape, dtype=bl.cf32, device=bl.cpu)
    host_output = bl.array_tensor(output_shape, dtype=bl.cf32, device=bl.cpu)

    bl_input = host_input.as_numpy()
    bl_phasors = host_phasors.as_numpy()
    bl_output = host_output.as_numpy()

    np.copyto(bl_input, np.random.random(size=input_shape)+1j*np.random.random(size=input_shape))
    np.copyto(bl_phasors, np.random.random(size=phasor_shape)+1j*np.random.random(size=phasor_shape))

    #
    # Blade Implementation
    #

    pipeline = Pipeline(input_shape, phasor_shape, output_shape, config)
    pipeline(host_input, host_phasors, host_output)

    #
    # Python Implementation
    #

    py_output = np.zeros(output_shape, dtype=np.complex64)

    for ibeam in range(bl_phasors.shape[0]):
        phased = bl_input * bl_phasors[ibeam][..., :]
        py_output[ibeam] = phased.sum(axis=0)
    phased = bl_input * bl_phasors[-1][..., :]
    phased = (phased.real * phased.real) + (phased.imag * phased.imag)
    py_output[-1] = np.sqrt(phased.sum(axis=0))

    #
    # Compare Results
    #

    assert np.allclose(bl_output[:-1, :, :, :], py_output[:-1, :, :, :], rtol=0.01)
    assert np.allclose(bl_output[-1, :, :, :], py_output[-1, :, :, :], atol=250)
    print("Test successfully completed!")