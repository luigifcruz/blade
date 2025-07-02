import numpy as np
import sys
import matplotlib.pyplot as plt


class KurtosisMaskReader:
    def __init__(self, fpath, kbsize = 256, nants = 28):
        self.fpath = fpath
        self.bksize = 256
        self.nants = nants

    def proc_block(self, data, block_size, block_start_ind):
        block = np.zeros(shape = (block_size * 8)) - 1
        # block = np.zeros(shape = (self.nants, 192, 8192 / self.bksize, 2))

        data_sub_block = data[block_start_ind * block_size:(block_start_ind + 1) * block_size]
        for idx, val in enumerate(data_sub_block):
            # self.mask[idx * 8 : (idx + 1) * 8] = [(val & (2 ** p)) >> p for p in range(0, 8)]
            for p in range(0, 8):
                masked = (val & (2 ** p)) >> p
                block[idx * 8 + p] = masked

        block = block.reshape((self.nants, 192, int(8192 / self.bksize), 2))

        return block

    def read(self):
        f = open(self.fpath, "rb")
        data = f.read()
        self.rawdata = data

        rawblocksize = self.nants * 192 * (8192 / int(self.bksize * 8)) * 2

        assert int(rawblocksize) == rawblocksize
        rawblocksize = int(rawblocksize)

        nblocks = int(len(data) / rawblocksize)

        assert len(data) / rawblocksize == nblocks

        self.mask = np.zeros(shape = (self.nants, 192, nblocks * int(8192 / self.bksize), 2))
        f.close()

        for i in range(nblocks):
            block = self.proc_block(self.rawdata, rawblocksize, i)
            tstart = i * int(8192 / self.bksize)
            tend = (i + 1) * int(8192 / self.bksize)
            self.mask[0:self.nants, 0:192, tstart:tend, 0:2] = block
