---
title: Beamformer
description: Forms coherent beams from antenna voltages and phasors.
order: 51
category: Blade
---

The Beamformer is the core compute block of a BLADE beamforming pipeline. It forms coherent beams from a phased array by weighting each antenna voltage with a per-beam phasor, usually produced by the [Phasor](/docs/block-phasor) block, and summing over antennas. An incoherent beam built from the summed antenna powers can be appended after the coherent beams.

## How it works

Each compute cycle launches a runtime-compiled CUDA kernel with one thread block per channel and time slice. The kernel first caches all beam phasors in shared memory and the antenna samples of its time slot in registers, then multiplies each antenna voltage by its beam phasor and accumulates the products over antennas, independently per polarization. When the incoherent beam is enabled, the kernel additionally detects the power of each antenna after applying the phasors of the first beam and accumulates it into one extra beam at the last index, optionally taking its square root to convert power into amplitude.

## Configuration

| Parameter | Type | Default | Description |
|---|---|---|---|
| `enableIncoherentBeam` | boolean | `false` | Append an incoherent beam after the coherent beams. |
| `enableIncoherentBeamSqrt` | boolean | `false` | Apply a square root to the incoherent beam power. |
| `blockSize` | integer | `512` | CUDA threads per block. The time axis must be divisible by this value and it must be at least the number of beams. |

## Input

| Name | Description |
|---|---|
| `buffer` | Contiguous `CF32` tensor shaped `[antennas, channels, samples, polarizations]` with exactly two polarizations. |
| `phasors` | Contiguous `CF32` tensor shaped `[beams, antennas, channels, 1, polarizations]` matching the input antenna and channel counts. |

## Output

| Name | Description |
|---|---|
| `buffer` | `CF32` tensor shaped `[beams, channels, samples, polarizations]`. The incoherent beam, when enabled, is the last beam. |
