---
title: Detector
description: Integrates detected power products.
order: 55
category: Blade
---

The Detector converts dual-polarization complex voltages into detected power products and integrates them over time. It is the usual step between a [Beamformer](/docs/block-beamformer) and a spectrometer sink: with four output polarizations it produces XX, YY, and the real and imaginary parts of the cross product XY, and with one it produces the total power XX plus YY.

## How it works

Each compute cycle computes the power products of every dual-polarization sample and sums them into their integration windows. The CPU implementation performs one contiguous reduction per output sample. The CUDA implementation uses one thread per input sample and atomic accumulation. The time axis therefore shrinks by the integration rate while the polarization axis becomes the number of detected products, and the output turns real-valued.

## Configuration

| Parameter | Type | Default | Description |
|---|---|---|---|
| `integrationRate` | integer | `1` | Number of time samples summed into each output sample. The time axis must be divisible by this value. |
| `numberOfOutputPolarizations` | integer | `4` | Detected products per sample, `1` for total power or `4` for full products. |
| `blockSize` | integer | `512` | CUDA threads per block. Ignored on CPU. |

## Input

| Name | Description |
|---|---|
| `buffer` | Contiguous `CF32` tensor shaped `[antennas, channels, samples, polarizations]` with exactly two polarizations. |

## Output

| Name | Description |
|---|---|
| `buffer` | `F32` tensor shaped `[antennas, channels, samples / integrationRate, numberOfOutputPolarizations]`. |
