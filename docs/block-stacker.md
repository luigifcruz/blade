---
title: Stacker
description: Stacks the input along a specified axis.
order: 57
category: Blade
---

The Stacker tiles successive input buffers along a tensor axis, assembling an output whose axis is `ratio` times larger than the input. It is the standard way to batch short buffers into the larger block geometry expected by writers, correlators, or waterfall displays.

## How it works

The output is zeroed when a new stacking cycle begins. Each compute cycle then writes the current input buffer into its slot along the chosen axis, cycling through `ratio` slots, and the assembled output is only emitted once every slot has been filled. In between, the block skips downstream processing. The copy path is chosen by the width of the contiguous chunk being moved: narrow chunks use a runtime-compiled CUDA kernel, while wide chunks use a strided device memcopy.

## Configuration

| Parameter | Type | Default | Description |
|---|---|---|---|
| `axis` | integer | `0` | The tensor axis to stack along. |
| `ratio` | integer | `1` | Number of buffers tiled into one output. Must be greater than one. |
| `copySizeThreshold` | integer | `512` | Chunk width in elements below which the kernel path is used instead of a strided memcopy. |
| `blockSize` | integer | `512` | CUDA threads per block for the kernel path. |

## Input

| Name | Description |
|---|---|
| `buffer` | Contiguous `CF32` tensor of any rank greater than the chosen axis. |

## Output

| Name | Description |
|---|---|
| `buffer` | `CF32` tensor with the chosen axis multiplied by `ratio` and all other axes unchanged. |
