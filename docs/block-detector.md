---
title: Detector
description: Integrates detected power products.
order: 55
category: Blade
---

The Detector converts dual-polarization complex voltages into detected power products and integrates them over time. It is the usual step between a [Beamformer](/docs/block-beamformer) and a spectrometer sink: with four output polarizations it produces XX, YY, and the real and imaginary parts of the cross product XY, and with one it produces the total power XX plus YY.

## How it works

Each compute cycle zeroes the output buffer and launches one of two precompiled CUDA kernels, selected by the number of output polarizations. One thread handles each dual-polarization input sample, computes its power products, and atomically accumulates them into the output sample of its integration window. The time axis therefore shrinks by the integration rate while the polarization axis becomes the number of detected products, and the output turns real-valued.

## Configuration

| Parameter | Type | Default | Description |
|---|---|---|---|
| `integrationRate` | integer | `1` | Number of time samples summed into each output sample. The time axis must be divisible by this value. |
| `numberOfOutputPolarizations` | integer | `4` | Detected products per sample, `1` for total power or `4` for full products. |
| `blockSize` | integer | `512` | CUDA threads per block. |

## Input

| Name | Description |
|---|---|
| `buffer` | Contiguous `CF32` tensor shaped `[antennas, channels, samples, polarizations]` with exactly two polarizations. |

## Output

| Name | Description |
|---|---|
| `buffer` | `F32` tensor shaped `[antennas, channels, samples / integrationRate, numberOfOutputPolarizations]`. |
