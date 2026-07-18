---
title: Correlator
description: Correlates voltages into baseline visibilities.
order: 53
category: Blade
---

The Correlator is the compute block of a BLADE interferometry pipeline. It cross-correlates dual-polarization antenna voltages into baseline visibilities, producing the four polarization products XX, XY, YX, and YY for every antenna pair, including the autocorrelations. Successive input buffers can be accumulated into a single output with the integration rate, and the result pairs naturally with a stacked writer such as the [UVH5 Writer](/docs/block-uvh5-writer).

## How it works

Each compute cycle correlates all upper-triangular antenna pairs independently per channel. The CPU implementation stages one dual-polarization sample per antenna and accumulates all baselines with the selected intermediate type. The CUDA implementation stages antenna voltages in shared memory and reduces 2x2 baseline tiles in registers. The output is zeroed at the start of each integration window and is only emitted once `integrationRate` inputs have been accumulated. In between, the block skips downstream processing.

On CPU, `calculationMode` directly selects 32-bit integer, single-precision, or double-precision accumulation. On CUDA, eligible `CI8` input packs four consecutive time samples into each 32-bit word and uses the `dp4a` instruction; this packed path is exact integer arithmetic regardless of `calculationMode`. When packing is unavailable, `CI8` uses the configured calculation mode. `CF32` supports single-precision or double-precision calculation on both backends, while integer calculation is rejected.

Floating-point addition depends on order. If several CUDA blocks added partial results to the same visibility, their completion order could change the lowest output bits between runs. The correlator avoids that behavior so identical input and configuration produce bit-reproducible visibilities.

The packed `CI8` path can safely split the time axis because each block adds its partial result to an exact 32-bit integer accumulator. Integer addition does not depend on block completion order. After the complete input buffer has been reduced, a second kernel converts the result to `CF32` and adds it to the integration output once. The generic path, used by `CF32` and unpacked `CI8`, assigns the complete time axis to one block instead. This gives up time-axis parallelism but keeps the calculation order fixed. In both paths, each visibility receives one floating-point addition per input buffer, and the CUDA stream processes those buffers in order.

Both paths derive their launch geometry from the input shape. The antenna axis is padded to an even count, and the shared-memory stage is sized to the 48 KB static budget. The packed `CI8` path may split the time axis across several blocks. It requires the time axis to tile by four and to be at most 65535 samples. Otherwise `CI8` falls back to the generic path, which uses one time chunk.

## Support matrix

| Input condition | `integer` | `single_precision_fp` | `double_precision_fp` |
|---|---|---|---|
| `CI8`, packed eligible | Packed + accumulate: safe, CF32 rounding | Packed + accumulate: safe, CF32 rounding | Packed + accumulate: safe, CF32 rounding |
| `CI8`, packed unavailable and samples <= 65535 | Generic: safe, CF32 rounding | Generic: safe, CF32 rounding | Generic: safe, CF32 rounding |
| `CI8`, samples > 65535 | Rejected to prevent integer overflow | Generic: conditional on integration size | Generic: conditional on integration size |
| `CF32` | Rejected to prevent integer overflow | Generic: conditional on input magnitude | Generic: recommended, but CF32 output can still overflow |

Packed eligibility requires a time dimension divisible by four and no greater than 65535 samples, as well as launch geometry that fits the shared-memory budget. Safe means overflow-safe, not numerically exact after conversion to the CF32 output.

## Configuration

| Parameter | Type | Default | Description |
|---|---|---|---|
| `integrationRate` | integer | `1` | Number of input buffers accumulated into each output buffer. |
| `conjugateAntennaIndex` | integer | `1` | Which antenna of the pair is conjugated, `0` for antenna A and `1` for antenna B. |
| `calculationMode` | string | `double_precision_fp` | Intermediate calculation precision. See the support matrix for available modes. |

## Input

| Name | Description |
|---|---|
| `buffer` | Contiguous `CI8` or `CF32` tensor shaped `[antennas, channels, samples, polarizations]` with exactly two polarizations. |

## Output

| Name | Description |
|---|---|
| `buffer` | `CF32` tensor shaped `[baselines, channels, 1, 4]` where baselines is `antennas * (antennas + 1) / 2`. |
