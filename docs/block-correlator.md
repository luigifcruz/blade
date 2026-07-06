---
title: Correlator
description: Correlates voltages into baseline visibilities.
order: 53
category: Blade
---

The Correlator is the compute block of a BLADE interferometry pipeline. It cross-correlates dual-polarization antenna voltages into baseline visibilities, producing the four polarization products XX, XY, YX, and YY for every antenna pair, including the autocorrelations. Successive input buffers can be accumulated into a single output with the integration rate, and the result pairs naturally with a stacked writer such as the [UVH5 Writer](/docs/block-uvh5-writer).

## How it works

Each compute cycle launches a runtime-compiled CUDA kernel with one thread block per reference antenna and channel group. For every baseline the kernel multiplies the voltages of one antenna with the complex conjugate of the other, sums the products over the time axis, and atomically accumulates the four polarization products into the packed upper-triangular baseline layout. The intermediate multiply runs at the precision selected by the calculation mode while the accumulation is always single-precision floating point. The output is zeroed at the start of each integration window and the buffer is only emitted once `integrationRate` inputs have been accumulated. In between, the block skips downstream processing. When the time axis dominates the channel axis, the reference antenna voltages can optionally be cached in shared memory.

## Configuration

| Parameter | Type | Default | Description |
|---|---|---|---|
| `integrationRate` | integer | `1` | Number of input buffers accumulated into each output buffer. |
| `conjugateAntennaIndex` | integer | `1` | Which antenna of the pair is conjugated, `0` for antenna A and `1` for antenna B. |
| `useSharedMemory` | boolean | `false` | Cache the reference antenna voltages in shared memory when the time axis dominates. |
| `calculationMode` | string | `double_precision_fp` | Precision of the intermediate multiply: `integer`, `single_precision_fp`, or `double_precision_fp`. |
| `blockSize` | integer | `32` | CUDA threads per block. The channel or time axis must be divisible by this value. |

## Input

| Name | Description |
|---|---|
| `buffer` | Contiguous `CI8` or `CF32` tensor shaped `[antennas, channels, samples, polarizations]` with exactly two polarizations. |

## Output

| Name | Description |
|---|---|
| `buffer` | `CF32` tensor shaped `[baselines, channels, 1, 4]` where baselines is `antennas * (antennas + 1) / 2`. |
