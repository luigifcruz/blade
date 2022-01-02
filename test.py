import blade as bl
import numpy as np


class Test(bl.Pipeline):
    beamformer: bl.Beamformer
    input = bl.vector.cuda.cf32()
    phasors = bl.vector.cuda.cf32()

    def __init__(self, dims: bl.ArrayDims):
        bl.Pipeline.__init__(self)
        _config = bl.Beamformer.Config(dims, 512)
        _input = bl.Beamformer.Input(self.input, self.phasors)
        self.beamformer = self.connect(_config, _input)

    def inputSize(self):
        return self.beamformer.inputSize()

    def phasorSize(self):
        return self.beamformer.phasorSize()

    def outputSize(self):
        return self.beamformer.outputSize()

    def run(self, input: bl.vector.cpu.cf32,
                  phasor: bl.vector.cpu.cf32,
                  output: bl.vector.cpu.cf32):
        self.copy(self.beamformer.input(), input)
        self.copy(self.beamformer.phasor(), phasor)
        self.compute()
        self.copy(output, self.beamformer.output())
        self.synchronize()


dims = bl.ArrayDims(NBEAMS=16, NANTS=20, NCHANS=192, NTIME=8192, NPOLS=2)
mod = Test(dims)

input = bl.vector.cpu.cf32(mod.inputSize())
phasor = bl.vector.cpu.cf32(mod.phasorSize())
output = bl.vector.cpu.cf32(mod.outputSize())

a = np.array(input, copy=False)
a[:] = 1
print(a)

a = np.array(phasor, copy=False)
a[:] = 2

a = np.array(input, copy=False)
print(a)

mod.run(input, phasor, output)

a = np.array(output, copy=False)
print(a)
