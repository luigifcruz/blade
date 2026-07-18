---
title: Beamformer
description: Forms coherent beams from antenna voltages and phasors.
order: 51
category: Blade
---

The Beamformer is the core compute block of a BLADE beamforming pipeline. It forms coherent beams from a phased array by weighting each antenna voltage with a per-beam phasor, usually produced by the [Phasor](/docs/block-phasor) block, and summing over antennas. An incoherent beam built from the summed antenna powers can be appended after the coherent beams.

## How it works

Each compute cycle converts `CI8` samples to normalized single precision by dividing each component by 128, multiplies each antenna voltage by its beam phasor, and accumulates the products over antennas independently per polarization. The CPU implementation reuses a small antenna cache for each channel and time sample. The CUDA implementation uses a runtime-compiled kernel that caches phasors in shared memory and antenna samples in registers. When the incoherent beam is enabled, either backend additionally detects the power of each antenna after applying the phasors of the first beam and accumulates it into one extra beam at the last index, optionally taking its square root to convert power into amplitude.

## Configuration

| Parameter | Type | Default | Description |
|---|---|---|---|
| `enableIncoherentBeam` | boolean | `false` | Append an incoherent beam after the coherent beams. |
| `enableIncoherentBeamSqrt` | boolean | `false` | Apply a square root to the incoherent beam power. |
| `blockSize` | integer | `512` | CUDA threads per block. Ignored on CPU. On CUDA, the time axis must be divisible by this value and it must be at least the number of beams. |

## Input

| Name | Description |
|---|---|
| `buffer` | Contiguous `CI8` or `CF32` tensor shaped `[antennas, channels, samples, polarizations]` with exactly two polarizations. |
| `phasors` | Contiguous `CF32` tensor shaped `[beams, antennas, channels, 1, polarizations]` matching the input antenna and channel counts. |

## Output

| Name | Description |
|---|---|
| `buffer` | `CF32` tensor shaped `[beams, channels, samples, polarizations]`. The incoherent beam, when enabled, is the last beam. |
