import numpy as np


class ATA:

    def __init__(self, NBEAMS: int, NANTS: int, NCHANS: int, NTIME: int, NPOLS: int):
        self.NBEAMS = NBEAMS
        self.NANTS = NANTS
        self.NCHANS = NCHANS
        self.NTIME = NTIME
        self.NPOLS = NPOLS

        self.input_len = (self.NANTS * self.NCHANS * self.NTIME * self.NPOLS)
        self.input_dims = (self.NANTS, self.NCHANS, self.NTIME, self.NPOLS)
        _a = np.random.uniform(-int(2**16/2), int(2**16/2), self.input_len)
        _b = np.random.uniform(-int(2**16/2), int(2**16/2), self.input_len)
        _c = (_a + _b * 1j).astype(np.complex64)
        self.input = _c.reshape(self.input_dims)

        # simulate complex phasors
        self.phasors = np.zeros(shape=(self.NBEAMS, self.NANTS, self.NCHANS, self.NPOLS), dtype=np.complex64)
        self.phasors[:] = np.random.random(size=self.phasors.shape) + 1j*np.random.random(size=self.phasors.shape)

        # generate zeroed output
        self.output = np.zeros(shape=(self.NBEAMS, self.NCHANS, self.NTIME, self.NPOLS), dtype=np.complex64)

    def process(self):
        for ibeam in range(self.NBEAMS):
            phased = self.input * self.phasors[ibeam][..., np.newaxis, :]
            self.output[ibeam] = phased.sum(axis=0)

    def saveToFile(self):
        self.phasors.tofile("phasor.raw")
        self.input.tofile("input.raw")
        self.output.tofile("output.raw")

    def getInputData(self):
        return self.input

    def getPhasorsData(self):
        return self.phasors

    def getOutputData(self):
        return self.output
