import time
import blade as bl
import numpy as np

class Test(bl.Pipeline):
    beamformer: bl.Beamformer

    def __init__(self, inputDims, phasorDims):
        bl.Pipeline.__init__(self)
        self.input = bl.vector.cuda.cf32.ArrayTensor(inputDims)
        self.phasors = bl.vector.cuda.cf32.PhasorTensor(phasorDims)
        _config = bl.Beamformer.Config(True, True, 512)
        _input = bl.Beamformer.Input(self.input, self.phasors)
        self.beamformer = self.connect(_config, _input)

    def run(self, input: bl.vector.cpu.cf32.ArrayTensor,
                  phasors: bl.vector.cpu.cf32.PhasorTensor,
                  output: bl.vector.cpu.cf32.ArrayTensor):
        self.copy(self.beamformer.input(), input)
        self.copy(self.beamformer.phasors(), phasors)
        self.compute()
        self.copy(output, self.beamformer.output())
        self.synchronize()


if __name__ == "__main__":
    # Specify dimension of array.
    inputDims = bl.vector.ArrayShape(2, 192, 512, 2)
    phasorDims = bl.vector.PhasorShape(1, 2, 192, 1, 2)
    outputDims = bl.vector.ArrayShape(2, 192, 512, 2)

    # Initialize Blade pipeline.
    mod = Test(inputDims, phasorDims)

    # Generate test data with Python.
    _a = np.random.uniform(-int(2**8/2), int(2**8/2), len(inputDims))
    _b = np.random.uniform(-int(2**8/2), int(2**8/2), len(inputDims))
    _c = np.array(_a + _b * 1j).astype(np.complex64)
    input = _c.reshape(inputDims.shape)

    _a = np.zeros(phasorDims.shape, dtype=np.complex64)
    phasors = np.random.random(size=_a.shape) + 1j*np.random.random(size=_a.shape)

    output = np.zeros(outputDims.shape, dtype=np.complex64)

    # Import test data from Python to Blade.
    bl_input = bl.vector.cpu.cf32.ArrayTensor(inputDims)
    bl_phasors = bl.vector.cpu.cf32.PhasorTensor(phasorDims)
    bl_output = bl.vector.cpu.cf32.ArrayTensor(outputDims)

    np.copyto(np.array(bl_input, copy=False), input.flatten())
    np.copyto(np.array(bl_phasors, copy=False), phasors.flatten())
    np.copyto(np.array(bl_output, copy=False), output.flatten())

    # Beamform with Blade.
    start = time.time()
    mod.run(bl_input, bl_phasors, bl_output)
    print(f"Beamform with Blade took {time.time()-start:.2f} s.")

    # Beamform with Numpy.
    start = time.time()
    for ibeam in range(phasorDims.shape[0]):
        phased = input * phasors[ibeam][..., :]
        output[ibeam] = phased.sum(axis=0)
    phased = input * phasors[-1][..., :]
    phased = (phased.real * phased.real) + (phased.imag * phased.imag)
    output[-1] = np.sqrt(phased.sum(axis=0))
    print(f"Beamform with Numpy took {time.time()-start:.2f} s.")

    # Check both answers.
    bl_out = np.array(bl_output, copy=False).reshape(output.shape)
    py_out = output

    print(bl_out[0, 0, :16, 0])
    print(py_out[0, 0, :16, 0])

    assert np.allclose(bl_out[:-1, :, :, :], py_out[:-1, :, :, :], rtol=0.01)
    assert np.allclose(bl_out[-1, :, :, :], py_out[-1, :, :, :], atol=250)
    print("Test successfully completed!")
