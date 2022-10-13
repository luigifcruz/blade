from astropy import units as u
import numpy as np
import setigen as stg
import h5py
import argparse
import os
import time

parser = argparse.ArgumentParser(
    description="Synthesize multi-antenna Raw files",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "bfr5filepath",
    type=str,
    default=None,
    help="Optional filepath of BFR5 to synthesize for.",
)
parser.add_argument("-a", type=int, help="Number of antenna.", default=20)
parser.add_argument("-c", type=int, help="Number of channels per antenna.", default=128)
parser.add_argument("-t", type=int, help="Number of time-samples.", default=16384)
parser.add_argument("-p", type=int, help="Number of polarisations.", default=2)
parser.add_argument("-b", type=int, help="Number of bits.", default=8)
parser.add_argument("-s", type=int, help="Sample rate (MHz).", default=2048)
parser.add_argument("-f", type=int, help="Number of F-Engine channels.", default=1024)
parser.add_argument("-D", type=str, help="Output directory.", default=".")
parser.add_argument("-B", type=int, help="Blocks per file.", default=64)
parser.add_argument("-F", type=int, help="Number of files.", default=1)
parser.add_argument("-S", type=str, help="Output stem.", default="synthesized_input")
args = parser.parse_args()


n_ant = args.a
n_chan_perant = args.c
n_time = args.t
n_pols = args.p
n_bits = args.b
sample_rate = args.s * 1e6
output_file_stem = args.S
blocks_per_file = args.B

# Internally Calculated from here on

chan_bw = sample_rate / 2 * args.f
delays = np.array([int(i * 5e-9 * sample_rate) for i in range(n_ant)])  # as samples

synctime = int(time.time())

bfr = h5py.File(args.bfr5filepath, "r") if args.bfr5filepath is not None else None
if bfr is not None:
    output_file_stem = os.path.splitext(os.path.basename(args.bfr5filepath))[0]
    n_pols = bfr["diminfo"]["npol"][()]
    n_ant = bfr["diminfo"]["nants"][()]
    n_chan_perant = bfr["diminfo"]["nchan"][()]
    n_time = bfr["diminfo"]["ntimes"][()]

    synctime = int(bfr["delayinfo"]["time_array"][0])
    
    # TODO calculate actual phase_center beam delays (['obsinfo']['phase_center_ra/dec])
    delays = np.array(
        [int(d * 1e9 / sample_rate) for d in bfr["delayinfo"]["delays"][0, 0, :]]
    )  # as samples

output_file_stem = os.path.join(args.D, output_file_stem)

block_bytesize = (n_time * n_ant * n_chan_perant * n_pols * 2 * n_bits) // 8

antenna_array = stg.voltage.MultiAntennaArray(
    num_antennas=n_ant,
    sample_rate=sample_rate * u.Hz,
    fch1=6 * u.GHz,
    ascending=False,
    num_pols=2,
    delays=delays,
    seed=3141592,
)

for stream in antenna_array.bg_streams:
    stream.add_noise(v_mean=0, v_std=1)

    stream.add_constant_signal(
        f_start=6 * u.GHz + (n_chan_perant / 3 + 0.5) * chan_bw * u.Hz,
        drift_rate=-2 * u.Hz / u.s,
        level=0.002,
    )

digitizer = stg.voltage.RealQuantizer(target_fwhm=32, num_bits=8)

filterbank = stg.voltage.PolyphaseFilterbank(num_taps=8, num_branches=n_chan_perant * 2)

requantizer = stg.voltage.ComplexQuantizer(target_fwhm=32, num_bits=n_bits)

rvb = stg.voltage.RawVoltageBackend(
    antenna_array,
    digitizer=digitizer,
    filterbank=filterbank,
    requantizer=requantizer,
    start_chan=0,
    num_chans=n_chan_perant,
    block_size=block_bytesize,
    blocks_per_file=blocks_per_file,
    num_subblocks=32,
)

rvb.record(
    output_file_stem=output_file_stem,
    num_blocks=blocks_per_file*args.F,
    length_mode="num_blocks",
    header_dict={
        "TELESCOP": "SETIGEN",
        "OBSID": "SYNTHETIC",
        "SYNCTIME": synctime,
        "SCHAN": 0,
        "PIPERBLK": n_time,
    },
    verbose=False,
    load_template=False,
)

print(output_file_stem)