#!/usr/bin/env python3

# Copyright 2021 Daniel Estevez <daniel@destevez.net>
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord, AltAz, ICRS, ITRS
import astropy.units as u
import astropy.constants as const
import numpy as np


def compute_uvw(ts, source, ant_coordinates, ref_coordinates):
    """Computes UVW antenna coordinates with respect to reference

    Args:
        ts: array of Times to compute the coordinates
        source: source SkyCoord
        ant_coordinates: antenna ECEF coordinates.
             This is indexed as (antenna_number, xyz)
        ref_coordinates: phasing reference centre coordinates.
             This is indexed as (xyz)

    Returns:
        The UVW coordinates in metres of each of the baselines formed
        between each of the antennas and the phasing reference. This
        is indexed as (time, antenna_number, uvw)
    """
    baselines_itrs = ant_coordinates - ref_coordinates

    # Calculation of vector orthogonal to line-of-sight
    # and pointing due north.
    north_radec = [source.ra.deg, source.dec.deg + 90]
    if north_radec[1] > 90:
        north_radec[1] = 180 - north_radec[1]
        north_radec[0] = 180 + north_radec[0]
    north = SkyCoord(ra=north_radec[0]*u.deg, dec=north_radec[1]*u.deg)

    source_itrs = source.transform_to(ITRS(obstime=Time(ts))).cartesian
    north_itrs = north.transform_to(ITRS(obstime=Time(ts))).cartesian
    east_itrs = north_itrs.cross(source_itrs)

    ww = baselines_itrs @ source_itrs.xyz.value
    vv = baselines_itrs @ north_itrs.xyz.value
    uu = baselines_itrs @ east_itrs.xyz.value
    uvw = np.stack((uu.T, vv.T, ww.T), axis=-1)

    return uvw


def compute_antenna_gainphase(uvw, delays, freq, n_ch, ch_bw, const_phase):
    """Computes the gain phase correction for each antenna

    Args:
        uvw: UVW coordinates of the antenna with respect to the reference
             in metres, indexed as (time, antenna, uvw).
        delays: cable delays of each antenna in seconds, indexed as
             (time, antenna). These should be the residual delays in case
             any delay correction has been applied in the receiver.
        freq: sky frequency corresponding to the centre of the passband in Hz
        n_ch: number of frequency channels
        ch_bw: bandwidth or sample rate of each channel in Hz
        phase: phase of each antenna, in radians
            (antenna)

    Returns:
        The gain phase correction (as a complex number of modulus one)
        corresponding to each time and antenna, indexed as
        (time, antenna, channel)
    """
    w_seconds = uvw[..., 2] / const.c.value
    w_cycles = w_seconds * freq
    phase = np.exp(-1j*2*np.pi*w_cycles)[..., np.newaxis]

    delays = (delays - w_seconds)[..., np.newaxis]
    ch_idx = np.arange(-n_ch//2, n_ch//2)
    phase_slope = np.exp(1j*2*np.pi*delays*ch_idx*ch_bw)

    const_phase = np.array(const_phase)[np.newaxis, ..., np.newaxis]

    return phase * phase_slope * np.exp(1j*const_phase)


def compute_baseline_gainamplitude(w, delays, n_ch, ch_bw):
    """Computes the gain amplitude correction of a single baseline

    The (non-closing) gain amplitude correction compensates correlation
    losses due to non-overlapping samples in the correlation caused by
    uncorrected delays.

    Args:
        w: W coordinates of the baseline in metres, indexed as
             by time.
        delays: cable delays of the baseline, defined as the delay of
             antenna 1 minus the delay of antenna 2, indexed by time
        n_ch: number of frequency channels
        ch_bw: bandwidth or sample rate of each channel in Hz

    Returns:
        The gain amplitude correction corresponding to each time and
        baseline, indexed by time
    """
    w_seconds = w / const.c.value
    delays = delays - w_seconds
    amplitude_loss = np.abs(delays * ch_bw)
    return 1 / (1 - amplitude_loss)


def apply_phasing(ts, corrs, source, freq, ch_bw, ant_coordinates,
                  delays, ref_coordinates=None):
    """Applies in-place the phasing to the correlation matrix

    The correlation matrix is modified in place. The UVW coordinates are
    returned.

    Args:
        ts: array of Times to compute the coordinates
        source: source SkyCoord
        corrs: correlation matrix, indexed as
            (time, channel, baseline, polarization)
        freq: sky frequency corresponding to the centre of the passband in Hz
        ch_bw: bandwidth or sample rate of each channel in Hz
        ant_coordinates: antenna ECEF coordinates.
             This is indexed as (antenna_number, xyz)
        delays: cable delays of each antenna in seconds, either indexed as
             (time, antenna) or (antenna) if they are time-invariant. These
             should be the residual delays in case any delay correction has
             been applied in the receiver.
        ref_coordinates: phasing reference centre coordinates.
             This is indexed as (xyz). By default, the average of the antenna
             coordinates is taken.

    Returns:
        The UVW coordinates in metres of each of the baselines. This
        is indexed as (time, baseline, uvw)
    """
    if delays.ndim == 1:
        delays = np.repeat(delays[np.newaxis, :], ts.size, axis=0)

    if ref_coordinates is None:
        ref_coordinates = np.average(ant_coordinates, axis=0)

    uvw = compute_uvw(ts, source, ant_coordinates, ref_coordinates)
    uvw_baseline = np.zeros((ts.size, corrs.shape[2], 3))

    n_ch = corrs.shape[1]
    gainphase = compute_antenna_gainphase(
        uvw, delays, freq, n_ch, ch_bw)
    baselines = [(j, k) for j in range(ant_coordinates.shape[0])
                 for k in range(j+1)]
    for j, b in enumerate(baselines):
        if b[0] == b[1]:
            # No need to do anything for autocorrelations.
            # The corresponding uvw_baseline is already initialized
            # to zero.
            continue
        corrs[:, :, j, :] *= gainphase[:, b[0], :][..., np.newaxis]
        corrs[:, :, j, :] *= np.conjugate(
            gainphase[:, b[1], :][..., np.newaxis])
        uvw_baseline[:, j, :] = uvw[:, b[0]] - uvw[:, b[1]]
        gainamp = compute_baseline_gainamplitude(
            uvw_baseline[:, j, 2], delays[:, b[0]] - delays[:, b[1]],
            n_ch, ch_bw)
        corrs[:, :, j, :] *= gainamp[:, np.newaxis, np.newaxis]

    return uvw_baseline


def midpoint_timestamps(t0, n, T):
    """Returns the timestamps corresponding to the midpoint of each interval

    Args:
        t0: initial timestamp (Time)
        n: total number of timestamps (int)
        T: interval duration (seconds)

    Returns:
        An array with the Time's t0 + T/2, t0 + 3T/2, ...
    """
    T = TimeDelta(T, format='sec')
    return t0 + T/2 + np.arange(n) * T
