---
title: Integrator
description: Integrates (sums) samples along an axis.
order: 56
category: Blade
---

The Integrator sums samples along a tensor axis to raise the signal-to-noise ratio or reduce the data rate of a pipeline. It offers two mechanisms that can be combined: `size` sums groups of consecutive indices inside a single buffer, shrinking the axis, while `rate` accumulates successive buffers into the same output before it is emitted.

## How it works

Each accumulation cycle starts by zeroing the output buffer. A runtime-compiled CUDA kernel then sums `size` consecutive groups along the chosen axis and adds the result into the output. When `rate` is greater than one, the output is only emitted after that many buffers have been accumulated, and the block skips downstream processing in between. Setting both `size` and `rate` to one bypasses integration. The input can be real float, complex float, or complex signed byte, and the output is always complex float.

## Configuration

| Parameter | Type | Default | Description |
|---|---|---|---|
| `size` | integer | `1` | Number of indices summed together within one buffer along the axis. The axis must be divisible by this value. |
| `rate` | integer | `1` | Number of successive buffers accumulated into one output. |
| `axis` | integer | `2` | The tensor axis to integrate on, defaulting to the time axis. |
| `blockSize` | integer | `512` | CUDA threads per block. |

## Input

| Name | Description |
|---|---|
| `buffer` | Contiguous `F32`, `CF32`, or `CI8` tensor of any rank greater than the chosen axis. |

## Output

| Name | Description |
|---|---|
| `buffer` | `CF32` tensor with the chosen axis divided by `size` and all other axes unchanged. |
