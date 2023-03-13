import time
from random import random
import blade as bl
import numpy as np

class Test(bl.Pipeline):
    detector: bl.Detector

    def __init__(self, input_dims, detector_config: bl.Detector.Config):
        bl.Pipeline.__init__(self)
        self.input = bl.vector.cuda.cf32.ArrayTensor(input_dims)
        _config = detector_config
        _input = bl.Detector.Input(self.input)
        self.detector = self.connect(_config, _input)

    def run(self, input: bl.vector.cpu.cf32,
                  output: bl.vector.cpu.f32):
        self.copy(self.detector.input(), input)
        self.compute()
        self.copy(output, self.detector.output())
        self.synchronize()


if __name__ == "__main__":
    NTIME = 8750
    TFACT = 10
    NCHANS = 192
    OUTPOLS = 1
    NBEAMS = 2
    NPOLS = 2

    input_dims = bl.vector.ArrayShape(NBEAMS, NCHANS, NTIME, NPOLS)
    output_dims = bl.vector.ArrayShape(NBEAMS, NCHANS, NTIME // TFACT, OUTPOLS)

    #
    # Blade Implementation
    #

    detector_config = bl.Detector.Config(
        integration_size = TFACT,
        number_of_output_polarizations = OUTPOLS,
        
        block_size = 512
    )

    mod = Test(input_dims, detector_config)

    bl_input_raw = bl.vector.cpu.cf32.ArrayTensor(input_dims)
    bl_output_raw = bl.vector.cpu.f32.ArrayTensor(output_dims)

    bl_input = np.array(bl_input_raw, copy=False).reshape(input_dims.shape)
    bl_output = np.array(bl_output_raw, copy=False).reshape(output_dims.shape)

    np.copyto(bl_input, np.random.random(size=bl_input.shape) + 1j*np.random.random(size=bl_input.shape))

    start = time.time()
    for _ in range(10):
        mod.run(bl_input_raw, bl_output_raw)
    print(f"Detection with Blade took {time.time()-start:.2f} s.")

    #
    # Python Implementation
    #

    py_output = np.zeros((NBEAMS, NCHANS, NTIME//TFACT, OUTPOLS), dtype=np.float32)
    
    start = time.time()
    for ibeam in range(NBEAMS):
        for ichan in range(NCHANS):
            for isamp in range(NTIME//TFACT):
                x = bl_input[ibeam, ichan, isamp*TFACT:isamp*TFACT+TFACT, 0] #just to make code more visible
                y = bl_input[ibeam, ichan, isamp*TFACT:isamp*TFACT+TFACT, 1] #just to make code more visible

                auto_x = x.real * x.real + x.imag * x.imag
                auto_y = y.real * y.real + y.imag * y.imag

                py_output[ibeam, ichan, isamp, 0] = np.sum(auto_x) + np.sum(auto_y)
    print(f"Detection with Python took {time.time()-start:.2f} s.")

    #
    # Compare Results
    #

    assert np.allclose(bl_output, py_output, rtol=0.1)
    print("Test successfully completed!")
