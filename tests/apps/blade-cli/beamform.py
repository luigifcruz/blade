import os, re, argparse
import numpy
import pyproj
import time

import astropy.constants as const
from astropy.coordinates import SkyCoord
from astropy.time import Time

import erfa

def upchannelize_frequencies(frequencies, rate):
    fine_frequencies = numpy.zeros((len(frequencies), rate), dtype=numpy.float64)
    chan_bw = 0
    for coarse_chan_i in range(len(frequencies)-1):
        chan_bw = frequencies[coarse_chan_i+1] - frequencies[coarse_chan_i]
        fine_frequencies[coarse_chan_i, :] = numpy.linspace(
            frequencies[coarse_chan_i],
            frequencies[coarse_chan_i+1],
            rate,
            endpoint=False
        )
    fine_frequencies[-1, :] = numpy.linspace(
        frequencies[-1],
        frequencies[-1]+chan_bw,
        rate,
        endpoint=False
    )
    return fine_frequencies.flatten()


def upchannelize(
    datablock: numpy.ndarray, # [Antenna, Frequency, Time, Polarization]
    rate: int
):
    """
    Params
    ------
        datablock: numpy.ndarray # [Antenna, Frequency, Time, Polarization]
            the data to be upchannelized
        frequencies: list
            the frequency of each channel, will be upchannelized and returned
        rate: int
            the FFT length
    
    Return
    ------
    upchannelized_datablock, upchannelized_frequencies
    """
    # Be Faster!
    A, F, T, P = datablock.shape
    assert T % rate == 0, f"Rate {rate} is not a factor of time {T}."
    fine_datablock = datablock.copy()
    fine_datablock = fine_datablock.reshape((A, F, T//rate, rate, P))
    fine_datablock = numpy.fft.fftshift(numpy.fft.fft(
            fine_datablock,
            axis=3
        ),
        axes=3
    )
    fine_datablock = numpy.transpose(fine_datablock, (0, 1, 3, 2, 4))
    return fine_datablock.reshape((A, F*rate, T//rate, P))

    # not slower!
    A, F, T, P = datablock.shape
    assert T % rate == 0, f"Rate {rate} is not a factor of time {T}."
    output = numpy.zeros((A, F*rate, T//rate, P), dtype=numpy.complex64)
    for iant in range(A):
        ant = datablock[iant]

        for ichan in range(F):
            time_pol = ant[ichan]

            for ipol in range(P):
                time_arr = time_pol[:, ipol]

                for ispec in range(output.shape[2]):
                    output[
                        iant,
                        ichan*rate : (ichan+1)*rate,
                        ispec,
                        ipol
                    ] = numpy.fft.fftshift(numpy.fft.fft(
                        time_arr[ispec*rate:(ispec+1)*rate]
                    ))

    return output

def _compute_ha_dec_with_astrom(astrom, radec):
    """Computes UVW antenna coordinates with respect to reference
    Args:
        radec: SkyCoord
    
    Returns:
        (ra=Hour-Angle, dec=Declination, unit='rad')
    """
    ri, di = erfa.atciq(
        radec.ra.rad, radec.dec.rad,
        0, 0, 0, 0,
        astrom
    )
    aob, zob, ha, dec, rob = erfa.atioq(
        ri, di,
        astrom
    )
    return ha, dec

def _compute_uvw(ts, source, ant_coordinates, lla, dut1=0.0):
    """Computes UVW antenna coordinates with respect to reference

    Args:
        ts: array of Times to compute the coordinates
        source: source SkyCoord
        ant_coordinates: numpy.ndarray
            Antenna XYZ coordinates, relative to reference position. This is indexed as (antenna_number, xyz)
        lla: tuple Reference Coordinates (radians)
            Longitude, Latitude, Altitude. The antenna_coordinates must have
            this component in them.

    Returns:
        The UVW coordinates in metres of each antenna. This
        is indexed as (antenna_number, uvw)
    """

    # get valid eraASTROM instance
    astrom, eo = erfa.apco13(
        ts.jd, 0,
        dut1,
        *lla,
        0, 0,
        0, 0, 0, 0
    )
    ha_rad, dec_rad = _compute_ha_dec_with_astrom(astrom, source)
    sin_long_minus_hangle = numpy.sin(lla[0]-ha_rad)
    cos_long_minus_hangle = numpy.cos(lla[0]-ha_rad)
    sin_declination = numpy.sin(dec_rad)
    cos_declination = numpy.cos(dec_rad)

    uvws = numpy.zeros(ant_coordinates.shape, dtype=numpy.float64)
    
    for ant in range(ant_coordinates.shape[0]):
        # RotZ(long-ha) anti-clockwise
        x = cos_long_minus_hangle*ant_coordinates[ant, 0] - (-sin_long_minus_hangle)*ant_coordinates[ant, 1]
        y = (-sin_long_minus_hangle)*ant_coordinates[ant, 0] + cos_long_minus_hangle*ant_coordinates[ant, 1]
        z = ant_coordinates[ant, 2]
        
        # RotY(declination) clockwise
        x_ = x
        x = cos_declination*x_ + sin_declination*z
        z = -sin_declination*x_ + cos_declination*z
        
        # Permute (WUV) to (UVW)
        uvws[ant, 0] = y
        uvws[ant, 1] = z
        uvws[ant, 2] = x
        
    return uvws


def phasors(
    antennaPositions: numpy.ndarray, # [Antenna, XYZ]
    boresightCoordinate: SkyCoord, # ra-dec
    beamCoordinates: 'list[SkyCoord]', #  ra-dec
    times: numpy.ndarray, # [unix]
    frequencies: numpy.ndarray, # [channel-frequencies] Hz
    calibrationCoefficients: numpy.ndarray, # [Frequency-channel, Polarization, Antenna]
    lla: tuple, # Longitude, Latitude, Altitude (radians)
    referenceAntennaIndex: int = 0,
):
    """
    Return
    ------
        phasors (B, A, F, T, P), delays_ns (T, B, A)

    """

    assert frequencies.shape[0] % calibrationCoefficients.shape[0] == 0, f"Calibration Coefficients' Frequency axis is not a factor of frequencies: {calibrationCoefficients.shape[0]} vs {frequencies.shape[0]}."

    phasorDims = (
        beamCoordinates.shape[0],
        antennaPositions.shape[0],
        frequencies.shape[0],
        times.shape[0],
        calibrationCoefficients.shape[1]
    )
    calibrationCoeffFreqRatio = frequencies.shape[0] // calibrationCoefficients.shape[0]
    calibrationCoefficients = numpy.repeat(calibrationCoefficients, calibrationCoeffFreqRatio, axis=0) # repeat frequencies

    phasors = numpy.zeros(phasorDims, dtype=numpy.complex128)
    
    delays_ns = numpy.zeros(
        (
            times.shape[0],
            beamCoordinates.shape[0],
            antennaPositions.shape[0],
        ),
        dtype=numpy.float64
    )

    
    # TODO don't go back and forth... we're mimicing BLADE directly, but it shouldn't either
    transformer = pyproj.Proj.from_proj(
        pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84'),
        pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84'),
    )
    referenceXyz = transformer.transform(
        lla[0] * 180.0 / numpy.pi,
        lla[1] * 180.0 / numpy.pi,
        lla[2],
    )
    for i in range(antennaPositions.shape[0]):
        antennaPositions[i, :] -= referenceXyz

    index = 0
    t_start = time.time()
    for t, tval in enumerate(times):
        ts = Time(tval, format='unix')
        boresightUvw = _compute_uvw(
            ts,
            boresightCoordinate, 
            antennaPositions,
            lla
        )
        # UVWs are relative to reference antenna, 
        boresightUvw -= boresightUvw[referenceAntennaIndex:referenceAntennaIndex+1, :]
        for b in range(phasorDims[0]):
            beamUvw = _compute_uvw( # [Antenna, UVW]
                ts,
                beamCoordinates[b], 
                antennaPositions,
                lla
            )
            beamUvw -= beamUvw[referenceAntennaIndex:referenceAntennaIndex+1, :]

            delays_ns[t, b, :] = (beamUvw[:,2] - boresightUvw[:,2]) * (1e9 / const.c.value)
            for a, delay in enumerate(delays_ns[t, b, :]):                
                phasor = numpy.exp(
                    -1.0j*2.0*numpy.pi
                    *delay*1e-9
                    *frequencies
                )

                phasors[b, a, :, t, :] = numpy.reshape(numpy.repeat(phasor, 2), (len(phasor), 2)) * calibrationCoefficients[:, :, a]
                
                index += 1
                elapsed = time.time() - t_start
                completion = index/len(delays_ns)

    return phasors, delays_ns


def beamform(
    datablock: numpy.ndarray, # [Antenna, Frequency, Time, Polarization]
    phasors: numpy.ndarray, # [Beam, Aspect, Frequency, Time=1, Polarization]
    lastBeamIncoherent = False
):
    # Beamform with Numpy.
    output = numpy.multiply(datablock, phasors) 
    
    if lastBeamIncoherent:
        output[-1] = output[-1].real**2 + output[-1].imag**2

    output = output.sum(axis=1) # sum across antenna, collapsing that dimension

    if lastBeamIncoherent:
        output[-1] = numpy.sqrt(output[-1])
    return output


def guppiheader_get_blockdims(guppiheader):
    nof_obschan = guppiheader.get('OBSNCHAN', 1)
    nof_ant = guppiheader.get('NANTS', 1)
    nof_chan = nof_obschan // nof_ant
    nof_pol = guppiheader.get('NPOL', 1)
    nof_time = guppiheader.get('BLOCSIZE', 0) * 8 // (nof_ant * nof_chan * nof_pol * 2 * guppiheader.get('NBITS', 8))
    return (nof_ant, nof_chan, nof_time, nof_pol)

def guppiheader_get_unix_midblock(guppiheader):
    dims = guppiheader_get_blockdims(guppiheader)
    ntime = dims[2]
    synctime = guppiheader.get('SYNCTIME', 0)
    pktidx = guppiheader.get('PKTIDX', 0)
    tbin = guppiheader.get('TBIN', 1.0/guppiheader.get("CHAN_BW", 0.0))
    piperblk = guppiheader.get('PIPERBLK', ntime)
    return synctime + (pktidx + piperblk) * ((tbin * ntime)/piperblk)

def index_of_time(times, t):
    for i, ti in enumerate(times):
        if ti == t:
            return i
        if ti > t:
            assert i != 0, f"Time {t} appears to be before the start of times: {times[0]}"
            return i

    assert False, f"Time {t} appears to be past the end of times: {times[-1]}"


if __name__ == "__main__":
    import h5py
    from guppi.guppi import Guppi # https://github.com/MydonSolutions/guppi/tree/write

    parser = argparse.ArgumentParser(
        description='Generate a (RAW, BFR5):(Filterbank) input:output pair of beamforming',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('bfr5', type=str)
    parser.add_argument('guppi', type=str)

    parser.add_argument('-u', '--upchannelization-rate', type=int, default=1,)
    parser.add_argument('-P', '--postbeamform-upchannelize', action='store_true', help="As opposed to pre-beamformation upchannelization")
    parser.add_argument('-o', '--output-directory', type=str, default=None,)

    args = parser.parse_args()
    bfr5 = h5py.File(args.bfr5, 'r')
    gfile = Guppi(args.guppi)
    hdr, data = gfile.read_next_block()

    guppi_stem = re.match(r"(.*)\.\d{4}.raw", os.path.basename(args.guppi)).group(1)
    if args.output_directory is None:
        args.output_directory = os.path.dirname(args.guppi)
    
    bfr5Dims = (
        bfr5['diminfo']['nants'][()],
        bfr5['diminfo']['nchan'][()],
        bfr5['diminfo']['ntimes'][()],
        bfr5['diminfo']['npol'][()],
    )
    
    beamCoordinates = numpy.array([
        SkyCoord(
            bfr5['beaminfo']['ras'][beamIdx],
            bfr5['beaminfo']['decs'][beamIdx],
            unit='rad'
        )
        for beamIdx in range(bfr5['diminfo']['nbeams'][()])
    ])
    boresightCoordinate = SkyCoord(
        bfr5['obsinfo']['phase_center_ra'][()],
        bfr5['obsinfo']['phase_center_dec'][()],
        unit='rad'
    )

    antennaPositions = bfr5['telinfo']['antenna_positions'][:]
    antennaPositionFrame = bfr5['telinfo']['antenna_position_frame'][()].decode()

    # convert from antennaPositionFrame to 'ecef'
    assert antennaPositionFrame != 'enu'
    if antennaPositionFrame == 'xyz':
        print("Converting to ECEF from XYZ")
        transformer = pyproj.Proj.from_proj(
            pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84'),
            pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84'),
        )
        lla = (
            bfr5['telinfo']['longitude'][()],
            bfr5['telinfo']['latitude'][()],
            bfr5['telinfo']['altitude'][()],
        )
        referenceXyz = transformer.transform(*lla)
        print(f"Adding reference position to XYZ antenna-positions: {referenceXyz}")
        for i in range(antennaPositions.shape[0]):
            antennaPositions[i, :] += referenceXyz

    print("ECEF Antenna Positions (X, Y, Z):")
    for i in range(antennaPositions.shape[0]):
        print(f"\t{i}: ({antennaPositions[i, 0]}, {antennaPositions[i, 1]}, {antennaPositions[i, 2]})")

    print("Beam Coordinates (RA, DEC):")
    for i in range(beamCoordinates.shape[0]):
        print(f"\t{i}: ({', '.join(beamCoordinates[i].to_string(unit='rad').split(' '))})")


    frequencies = bfr5['obsinfo']['freq_array'][hdr['SCHAN']:] * 1e9
    times = bfr5['delayinfo']['time_array'][:]
    times_jd = bfr5['delayinfo']['jds'][:]

    if args.postbeamform_upchannelize:
        phasor_frequencies = frequencies
    else:
        phasor_frequencies = upchannelize_frequencies(
            frequencies,
            args.upchannelization_rate
        )
    
    phasor_frequencies += (phasor_frequencies[2] - phasor_frequencies[1])/2

    print(f"Generating Phasors...")
    print(phasor_frequencies*1e-9)
    phasorCoeffs, delays = phasors(
        antennaPositions, # [Antenna, XYZ]
        boresightCoordinate, # degrees
        beamCoordinates, #  degrees
        times, # [unix]
        phasor_frequencies, # [channel-frequencies] Hz
        bfr5['calinfo']['cal_all'][hdr['SCHAN']:hdr['SCHAN']+hdr['OBSNCHAN']//hdr['NANTS']], # [Antenna, Frequency-channel, Polarization]
        referenceAntennaIndex = 0,
        lla = (
            numpy.pi * bfr5['telinfo']['longitude'][()] / 180.0,
            numpy.pi * bfr5['telinfo']['latitude'][()] / 180.0,
            bfr5['telinfo']['altitude'][()],
        )
    )

    # for b in range(delays.shape[1]):
    #     for a in range(delays.shape[2]):
    #         print(f"Delay [b={b}, a={a}, t={times_jd[0]}: {delays[0, b, a]} ns")
    # for b in range(phasorCoeffs.shape[0]):
    #     for a in range(phasorCoeffs.shape[1]):
    #         for f in range(phasorCoeffs.shape[2]):
    #             if f >= 3 and f < phasorCoeffs.shape[2]-3:
    #                 continue
    #             for p in range(phasorCoeffs.shape[4]):
    #                 print(f"Phasor [b={b}, a={a}, f={f}, p={p}] @ {times_jd[0]} @ {phasor_frequencies[f]}: {phasorCoeffs[b,a,f,0,p]}")

    bfr5_delays = bfr5['delayinfo']['delays'][:]
    recipe_delays_agreeable = numpy.isclose(bfr5_delays, delays, atol=0.0001)
    # recipe_delays_agreeable = bfr5_delays == delays
    if not recipe_delays_agreeable.all():
        print(f"The delays in the provided recipe file do not match the calculated delays:\n{recipe_delays_agreeable}")
        # exit(1)
        print(f"Using calculated delays:\n{delays}")
        print(f"Not given delays:\n{bfr5_delays}")
    else:
        print("The recipe file's delays match the calculated delays.")


    block_index = 1
    block_times_index = 0
    file_open_mode = 'wb'
    while True:
        if hdr is None:
            break

        blockdims = guppiheader_get_blockdims(hdr)
        for dim in [0,2,3]:
            assert bfr5Dims[dim] == blockdims[dim], f"#{block_index}: {bfr5Dims} != {blockdims}"

        if args.postbeamform_upchannelize:
            beam_inputdata = data
        else:
            beam_inputdata = upchannelize(
                data,
                args.upchannelization_rate
            )

        try:
            block_times_index = index_of_time(
                times,
                guppiheader_get_unix_midblock(hdr)
            )
        except:
            pass
        beams = beamform(
            beam_inputdata,
            phasorCoeffs[:, :, :, block_times_index:block_times_index+1, :]
        )

        if args.postbeamform_upchannelize:
            beams = upchannelize(
                beams,
                args.upchannelization_rate
            )
        
        hdr["DATATYPE"] = "FLOAT"
        hdr["TBIN"] *= args.upchannelization_rate
        hdr["CHAN_BW"] /= args.upchannelization_rate
        Guppi.write_to_file(
            os.path.join(args.output_directory, f"{guppi_stem}-beamformed.raw"),
            hdr,
            beams.astype(numpy.complex64),
            file_open_mode=file_open_mode
        )
        file_open_mode = 'ab'
        hdr, data = gfile.read_next_block()
        block_index += 1
