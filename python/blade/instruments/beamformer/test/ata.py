import numpy as np


class ATA:

    def __init__(self, NBEAMS: int, NANTS: int, NCHANS: int, NTIME: int, NPOLS: int):
        self.NBEAMS = NBEAMS
        self.NANTS = NANTS
        self.NCHANS = NCHANS
        self.NTIME = NTIME
        self.NPOLS = NPOLS

        # simulate 8bit numbers (int8_t + int8_t = int16_t)
        self.input_len = (self.NANTS * self.NCHANS * self.NTIME * self.NPOLS)
        self.input_dims = (self.NANTS, self.NCHANS, self.NTIME, self.NPOLS)
        self.input_flat = np.random.randint(-int(2**16/2), int(2**16/2), self.input_len).astype(np.int16)

        # simulate complex phasors
        self.phasors = np.zeros(shape=(self.NBEAMS, self.NANTS, self.NCHANS, self.NPOLS), dtype=np.complex64)
        self.phasors[:] = np.random.random(size=self.phasors.shape) + 1j*np.random.random(size=self.phasors.shape)

        # convert int8 to complex64
        self.input = np.array(self.input_flat.view(np.int8)[::2] + 1j*self.input_flat.view(np.int8)[1::2],
                              dtype=np.complex64)
        self.input = self.input.reshape(self.input_dims)

        # generate zeroed output
        self.output = np.zeros(shape=(self.NBEAMS, self.NCHANS, self.NTIME, self.NPOLS), dtype=np.complex64)

    def process(self):
        for ibeam in range(self.NBEAMS):
            phased = self.input * self.phasors[ibeam][..., np.newaxis, :]
            self.output[ibeam] = phased.sum(axis=0)

    def saveToFile(self):
        self.phasors.tofile("phasor.raw")
        self.input_flat.tofile("input.raw")
        self.output.tofile("output.raw")

    def getInputData(self):
        return self.input_flat

    def getPhasorsData(self):
        return self.phasors

    def getOutputData(self):
        return self.output
