---
title: Polarizer
description: Reorients the polarizations.
order: 54
category: Blade
---

The Polarizer converts a dual linear-polarization complex signal into a circular basis or extracts a single linear component. The circular conversion forms the left beam as X plus jY and the right beam as X minus jY, which is the standard convention for observing circularly polarized sources with a linear feed.

## How it works

The block selects one of three runtime-compiled CUDA kernels based on the requested output polarization. The circular kernel multiplies the Y polarization by a ninety degree phasor and writes the sum and difference with the X polarization as the left and right circular components. The single-component kernels simply gather every X or Y sample into a tensor with a single polarization axis. The conversion is a pure per-sample operation, so every other axis of the input passes through unchanged.

## Configuration

| Parameter | Type | Default | Description |
|---|---|---|---|
| `inputPolarization` | string | `xy` | Polarization basis of the input signal. Only `xy` is supported. |
| `outputPolarization` | string | `lr` | Output basis: `lr` for circular, or `x` or `y` for a single linear component. |
| `blockSize` | integer | `512` | CUDA threads per block. |

## Input

| Name | Description |
|---|---|
| `buffer` | Contiguous `CF32` tensor shaped `[antennas, channels, samples, polarizations]` with exactly two polarizations. |

## Output

| Name | Description |
|---|---|
| `buffer` | `CF32` tensor with the same shape as the input for `lr`, or with a single polarization for `x` and `y`. |
