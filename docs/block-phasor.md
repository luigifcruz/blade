---
title: Phasor
description: Calculates the phasors for beamforming.
order: 52
category: Blade
---

The Phasor block computes the geometric delays and complex phasor weights that steer the [Beamformer](/docs/block-beamformer). From the ECEF antenna positions, the boresight pointing, and the coordinates of each beam, it derives one delay per beam and antenna and synthesizes one calibrated phasor per beam, antenna, channel, and polarization. All angles are in radians and all frequencies are in Hertz.

## How it works

The block first converts the ECEF antenna positions into the array-centered XYZ frame defined by the array reference coordinates. Using the Julian date and DUT1 inputs it builds an astrometry context with [radiointerferometryc99](https://github.com/MydonSolutions/radiointerferometryc99), converts the boresight and beam coordinates from right ascension and declination to hour angle and declination, and projects the antenna positions towards each direction to obtain per-antenna delays relative to the reference antenna. The boresight delay is subtracted from each beam delay, and the resulting relative delays are converted into complex exponentials per frequency channel, including the fringe rate at the band start frequency. Each phasor is finally multiplied by the per-antenna calibration before being emitted.

## Configuration

| Parameter | Type | Default | Description |
|---|---|---|---|
| `observationFrequencyHz` | float | `1.4e9` | Center frequency of the observation in hertz. |
| `channelBandwidthHz` | float | `1e6` | Bandwidth of a single frequency channel in hertz. |
| `totalBandwidthHz` | float | `1e6` | Total bandwidth of the observation in hertz. |
| `frequencyStartIndex` | integer | `0` | Zero-based index of the first frequency channel being processed. |
| `referenceAntennaIndex` | integer | `0` | Antenna used as the delay reference. |
| `arrayReferenceLongitude` | float | ATA | Longitude of the array reference position in radians. |
| `arrayReferenceLatitude` | float | ATA | Latitude of the array reference position in radians. |
| `arrayReferenceAltitude` | float | ATA | Altitude of the array reference position in meters. |

## Input

| Name | Description |
|---|---|
| `antennaPositions` | `F64` tensor shaped `[antennas, 3]` with the ECEF position of each antenna. |
| `antennaCalibrations` | `CF64` tensor shaped `[antennas, channels, polarizations]` with the per-antenna bandpass calibration. |
| `boresightCoordinates` | `F64` tensor with the boresight right ascension and declination. |
| `beamCoordinates` | `F64` tensor shaped `[beams, 2]` with the right ascension and declination of each beam. |
| `julianDate` | Scalar `F64` tensor with the Julian date of the observation. |
| `dut1` | Scalar `F64` tensor with the UT1 minus UTC correction. |

## Output

| Name | Description |
|---|---|
| `delays` | `F64` tensor shaped `[beams, antennas]` with the relative geometric delay in seconds. |
| `phasors` | `CF32` tensor shaped `[beams, antennas, channels, polarizations]` with the calibrated beamforming weights. |
