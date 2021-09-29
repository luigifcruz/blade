import numpy as np
import matplotlib.pyplot as plt
import glob
from shutil import copyfile as cp

from phasing import compute_uvw, compute_antenna_gainphase
from astropy.coordinates import ITRS, SkyCoord, AltAz, EarthLocation
from astropy.time import Time,TimeDelta
import pandas as pd

from guppi import guppi


itrf = pd.read_csv("./ant_itrf.txt", names=['x', 'y', 'z'], header=None, skiprows=1)

delays = [-1.40625e-07, -1.71875e-07, 0, -3.4375e-07, -9.84375e-07, -2.3437499999999998e-07, 
          -3.9062499999999997e-07, -7.65625e-07, -2.65625e-07, -1.65625e-06, -1.875e-07, -2.3437499999999998e-07]
sub_sample = [7.405235161218154e-09, -4.2500573407063695e-09, 
              0.0, 5.165503138214902e-10, -3.609053923420966e-09, 
              7.4443510061422865e-09, 1.7994035729740416e-09, 1.6197495652348848e-08, 
              -4.677579366160534e-09, -6.830792087152844e-09, -4.021061232869816e-09, 1.8680851926624956e-09]
phase = - np.array([-0.9703193924039318, 2.822218890772623, 0.0, 
         2.5113155068885518, -0.12355199010642844, 
         -3.1368145504317564, -0.7489464334015185, 
         -6.908397372665278, -1.7157150349036336, 0.3911021195915143, 
         -0.9675720293984017, -1.311942312625807])

delays = [i-j for i,j in zip(delays, sub_sample)]

source_ra, source_dec = 3.55, 54.579
ra  = source_ra * 360 / 24.
dec = source_dec
source = SkyCoord(ra, dec, unit="deg")

antnames = ['1A', '1F', '1C', '2A', '4J', '2H', '3D', '5B', '1K', '5C', '1H', '2B']
itrf_sub = itrf.loc[antnames]
refant = "1C"
irefant = itrf_sub.index.values.tolist().index(refant)

ofile_name = "coherent.fil"
cp("test.hdr", ofile_name)

f = open(ofile_name, "ab")
gnames = glob.glob("./buf0/J0332+5434/*.raw")

for gname in gnames:
    print(gname)
    g = guppi.Guppi(gname)
    for i in range(128):
        hdr, data = g.read_next_block()
        
        t_start = hdr['SYNCTIME'] + hdr['PKTSTART']*float(hdr['TBIN'])
        n_time_per_block = float(hdr['TBIN'])*hdr['PIPERBLK']
        print(i, t_start)
        
        ts = Time(t_start + n_time_per_block/2., format="unix")
        uvw = compute_uvw(ts, source, itrf_sub[['x','y','z']], itrf_sub[['x','y','z']].values[irefant])
        
        gain_phase = compute_antenna_gainphase(uvw, delays, hdr['OBSFREQ']*1e6, 256, 0.25*1e6, phase)
        gain_phase = gain_phase[0][..., np.newaxis, np.newaxis]
        
        d = data*gain_phase
        
        dd = d[[0,1,2,3,4,5,6,8,9,10,11]].sum(axis=0)
        dd = np.abs(dd).sum(axis=-1).T
        dd.astype("float32").tofile(f)
