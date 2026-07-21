---
title: Channelizer
description: Splits coarse-frequency voltages into centered fine-frequency channels.
order: 58
category: Blade
---

The Channelizer converts time-domain samples within each coarse-frequency channel into centered fine-frequency bins. It is a plain FFT channelizer rather than a polyphase filter bank: it applies no window, FIR taps, overlap, or normalization.

## How it works

The block multiplies alternating time samples by `-1` and performs a forward FFT over the time axis. For an even transform length, this produces the same bin ordering as `fftshift(fft(...))`. A zero-copy reshape then merges the coarse-channel and FFT-bin dimensions while retaining BLADE's `[aspects, channels, samples, polarizations]` convention.

The CPU path uses PocketFFT and the CUDA path uses cuFFT through CyberEther's FFT module. The transform length is inferred from the input sample dimension, which must be even unless it contains a single sample.

## Input

| Name | Description |
|---|---|
| `buffer` | Contiguous `CF32` tensor shaped `[aspects, coarse channels, samples, polarizations]` with one or two polarizations. |

## Output

| Name | Description |
|---|---|
| `buffer` | `CF32` tensor shaped `[aspects, coarse channels * samples, 1, polarizations]`. Fine bins are centered and grouped in coarse-channel-major order. |
